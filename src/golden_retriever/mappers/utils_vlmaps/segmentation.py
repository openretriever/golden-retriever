from collections import OrderedDict

import torch
import torch.nn as nn
from numpy import ndarray
from torch import Tensor
from torchvision import transforms

from retriever.mappers.utils_vlmaps.utils.lseg.modules.models.lseg_net import LSegEncNet


class Segmentation(nn.Module):
    def __init__(self, cfg):
        """Pretrained Visual-Language Model (VLM) used to extract pixel-level visual-language features"""
        super(Segmentation, self).__init__()

        """ Configurations """
        self.cfg = cfg

        """ Transformation """
        self.device = torch.device(cfg["VLM_MODEL"]["DEVICE"])
        self.norm_mean = cfg["VLM_MODEL"]["TRANSFORM"]["INPUT_MEAN"]
        self.norm_std = cfg["VLM_MODEL"]["TRANSFORM"]["INPUT_STD"]
        self.transform = transforms.Compose(
            [transforms.Normalize(self.norm_mean, self.norm_std)]
        )

        """ Default objects """
        self.lang_labels = cfg["VLM_MODEL"]["OBJ_LABELS"]

        """ Load LSeg Model """
        self.lseg_model = self.load_lseg_model(cfg["VLM_MODEL"]["MODEL_DIR"])

    def load_lseg_model(self, save_model_dir):
        """Load the pretrained LSeg model
        Set USE_TEXT_ENC to be True if you want to enable the CLIP text encoder
        """
        # create the LSeg model
        model = LSegEncNet(
            labels=self.lang_labels,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
            load_text_encoder=self.cfg["VLM_MODEL"]["CLIP"]["USE_TEXT_ENC"],
        )

        # load the pre-trained model state dict
        pretrained_state_dict = torch.load(save_model_dir)["state_dict"]

        # check whether load the CLIP text encoder
        if not self.cfg["VLM_MODEL"]["CLIP"]["USE_TEXT_ENC"]:
            model_state_dict = model.state_dict()  # current model state dict
            # remove the "net." prefix and clip related keys from the pretrained state dict
            pretrained_state_dict = {
                k.lstrip("net."): v
                for k, v in pretrained_state_dict.items()
                if "clip_pretrained" not in k
            }
            model_state_dict.update(pretrained_state_dict)  # update the state dict
        else:
            # remove the "net." prefix from the pretrained state dict
            pretrained_state_dict = {
                k.lstrip("net."): v for k, v in pretrained_state_dict.items()
            }

        # load the updated state dict
        model.load_state_dict(OrderedDict(pretrained_state_dict))
        model.eval()
        print("VLM model is loaded!")
        return model.to(self.device)

    def process_image(self, image_tensor: Tensor):
        """Convert image to tensor"""
        image_tensor = image_tensor.permute(0, 3, 1, 2) / 255.0
        img_tensor = self.transform(image_tensor)
        return img_tensor.contiguous().to(self.device)

    @staticmethod
    def get_segmentation_labels(vis_feats: Tensor, txt_feats: Tensor) -> ndarray:
        """Compute segmentation score for each pixel"""
        bs, h, w, d = vis_feats.shape

        # normalize the features
        batch_vis_feats = vis_feats.reshape(bs, h * w, d).float()
        batch_vis_feats = batch_vis_feats / batch_vis_feats.norm(dim=-1, keepdim=True)
        batch_txt_feats = txt_feats.T.unsqueeze(dim=0).repeat(bs, 1, 1).float()
        batch_txt_feats = batch_txt_feats / batch_txt_feats.norm(dim=-1, keepdim=True)

        # compute the similarity using dot product
        score_maps = torch.bmm(batch_vis_feats, batch_txt_feats).reshape(bs, h, w, -1)

        # select the maximal label
        sem_labels = torch.argmax(score_maps, dim=3).cpu().numpy()

        return sem_labels

    def forward_visual_features(self, input_batch: Tensor) -> Tensor:
        """Compute pixel-wise visual-language features using the visual encoder"""
        # convert to tensor
        image_tensor = self.process_image(input_batch)

        # compute the features
        output = self.lseg_model.extract_img_features(image_tensor)

        # This is the numpy array Batch x H x W x feature_dim containing the feature for each pixel.
        output = output.permute(0, 2, 3, 1).contiguous()
        return output

    def forward_textual_features(self, lang: str = "") -> Tensor:
        """Compute textual features using the textural encoder"""
        if lang != "":
            lang = ",".join([lang] + self.lang_labels)

        # compute the textual features
        text_features = self.lseg_model.extract_text_features(lang)
        return text_features
