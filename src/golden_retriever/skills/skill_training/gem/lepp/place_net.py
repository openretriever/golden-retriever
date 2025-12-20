import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import numpy as np
import torch
import torch.nn.functional as F
from cliport.models.streams.two_stream_attention_lang_fusion import (
    TwoStreamAttentionLangFusionLatLEPP,
    TwoStreamAttentionLangFusionLatLEPPPostLinearMul,
)
from cliport.utils import utils
from lepp.clip_revised.clip import tokenize
from lepp.equ_unet import EquiUnet, EquiUnetCLIPPostLinearADD
from lepp.kernel_backbone_mlp_conv import (
    CropLanUnet,
    CropUnet,
    CropUnetLanUNetADD,
    CropUnetLanUNetCAT,
)
from lepp.parser import parse_instruction
from lepp.unet import ResUnet
from lepp.unet_lan import ResUnet as ResUnetLan
from lepp.unet_lan import (
    ResUnetCLIPHead,
    ResUnetCLIPHeadHuggingFace,
    ResUnetCLIPPostADD,
    ResUnetCLIPPostCat,
    ResUnetCLIPPostLinearADD,
    ResUnetCLIPPostLinearMul,
    ResUnetCLIPPostLinearMulSigmoid,
    ResUnetCLIPPostLinearMulSoft,
    ResUnetCLIPPostMul,
    ResUnetCLIPPriorLinearMul,
    ResUnetSigCLIPPostLinearADDSig,
    ResUnetWoLanCLIPPostLinearMul,
)
from lifter import Transitor


class Transport(torch.nn.Module):
    def __init__(
        self,
        cfg,
        device,
        preprocess=None,
        init=False,
        model_name="unet",
        kernel_name="unet",
        vlm_name="clip",
        crop_size=64,
        lan_kernel=True,
        dist="transporter",
        vlm_model=None,
        lan_emb_dim=512,
    ):
        super().__init__()
        self.dist = dist
        self.device = device
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = 64 // 2
        self.preprocess = preprocess
        self.n_rotations = cfg["train"]["n_rotations"]
        self.quotient = False
        out_channel = cfg["lepp"]["logit_out_channel"]
        self.out_channel = out_channel
        self.parse = cfg["lepp"]["enable_parse"]
        self.enable_steer = cfg["lepp"]["enable_steer"]
        self.task = cfg["train"]["task"]
        self.lan_emb_dim = lan_emb_dim
        self.vlm_model = vlm_model

        self.model_name = model_name
        self.kernel_name = kernel_name
        in_shape = (320, 160, 6)

        if model_name == "unetl" or model_name == "pre_unetl":
            print("scratch unetl")
            self.obs_net = ResUnetLan(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unet" or model_name == "pre_unet":
            print("scratch unet")
            self.obs_net = ResUnet(
                preprocess=utils.preprocess,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unetl-m" or model_name == "pre_unetl-m":
            print("scratch unetl")
            self.obs_net = ResUnetLan(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unetln":
            print("scratch unetln")
            self.obs_net = ResUnetLan(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "cliport-lat":
            print("obsnet: scratch cliport-lateral")
            stream_one_fcn = "plain_resnet_lat"
            stream_two_fcn = "clip_lingunet_lat"

            self.obs_net = TwoStreamAttentionLangFusionLatLEPP(
                stream_fcn=(stream_one_fcn, stream_two_fcn),
                in_shape=in_shape,
                n_rotations=1,
                preprocess=utils.preprocess,
                cfg=cfg,
                device=self.device,
                out_channel=out_channel,
            )

        elif model_name == "cliport-lat-score-vit-postLinearMul":
            print("obsnet: scratch cliport-lateral")
            stream_one_fcn = "plain_resnet_lat"
            stream_two_fcn = "clip_lingunet_lat"

            self.obs_net = TwoStreamAttentionLangFusionLatLEPPPostLinearMul(
                stream_fcn=(stream_one_fcn, stream_two_fcn),
                in_shape=in_shape,
                n_rotations=1,
                preprocess=utils.preprocess,
                cfg=cfg,
                device=self.device,
                out_channel=out_channel,
            )

        elif model_name == "cliport-similarity-head":
            print("obsnet: scratch cliport-similarity")
            patch_size, patch_stride = 20, 20
            lan_dim = 1024
            self.obs_net = ResUnetCLIPHead(
                preprocess=utils.preprocess,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
                lan_dim=lan_dim,
            ).to(device)

        elif model_name == "cliport-similarity-head-hugging":
            print("obsnet: scratch cliport-similarity")
            patch_size, patch_stride = 20, 20
            model_id = "openai/clip-vit-base-patch16"
            lan_dim = 512
            self.obs_net = ResUnetCLIPHeadHuggingFace(
                device=device,
                preprocess=utils.preprocess,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
                model_id=model_id,
                lan_dim=lan_dim,
            ).to(device)

        elif model_name == "unetl-score-vit-postAdd":
            print("scratch unetl")
            self.obs_net = ResUnetCLIPPostADD(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unetl-score-vit-postMul":
            print("scratch unetl")
            self.obs_net = ResUnetCLIPPostMul(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unetl-score-vit-postLinearAdd":
            print("scratch unetl")
            self.obs_net = ResUnetCLIPPostLinearADD(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unetlSig-score-vit-postLinearAddSig":
            print("scratch unetl")
            self.obs_net = ResUnetSigCLIPPostLinearADDSig(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unetl-score-vit-postLinearMul":
            print("scratch unetl")
            self.obs_net = ResUnetCLIPPostLinearMul(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unet-score-vit-postLinearMul":
            print("scratch unetl")
            self.obs_net = ResUnetWoLanCLIPPostLinearMul(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unetl-score-vit-priorLinearMul":
            print("scratch unetl")
            self.obs_net = ResUnetCLIPPriorLinearMul(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unetl-score-vit-postLinearMulSigmoid":
            print("scratch unetl")
            self.obs_net = ResUnetCLIPPostLinearMulSigmoid(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unetl-score-vit-postLinearMul-m":
            print("unetl-score-vit-postLinearMul middle size")
            self.obs_net = ResUnetCLIPPostLinearMul(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(32, 64, 128, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unetl-score-vit-postLinearMulSoft":
            print("scratch unetl")
            self.obs_net = ResUnetCLIPPostLinearMulSoft(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)

        elif model_name == "unetl-score-vit-postCat":
            print("scratch unetl")
            self.obs_net = ResUnetCLIPPostCat(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)
            out_channel += 1

        elif model_name == "eunet-score-vit-postLinearAdd":
            print("scratch Eunet")
            print(init)
            self.obs_net = EquiUnetCLIPPostLinearADD(
                preprocess=utils.preprocess,
                in_dim=6,
                out_dim=out_channel,
                N=4,
                middle_dim=(16, 32, 64, 128),
                init=init,
            ).to(device)

        self.crop_size = crop_size
        lan_emd = True
        self.iters = 0

        # Padding the image to get 96*96 crop centered at pick location
        # self.crop_size_1 = 96
        self.crop_size_1 = 64
        self.pad_size_1 = int(self.crop_size_1 / 2)
        self.padding_1 = np.zeros((3, 2), dtype=int)
        self.padding_1[:2, :] = self.pad_size_1

        # override self.kernel_backbone
        if kernel_name == "unet":
            self.kernel_backbone = CropUnet(
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)
        elif kernel_name == "unet-lc-old":
            self.kernel_backbone = CropLanUnet(
                mlp_dim=(lan_emb_dim, 256, 128, 64),
                obs_in_channel=6,
                lan_in_channel=1,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)
        elif kernel_name == "unet-lc":
            mlp_dim = (1024, 256, 128, 64)
            lan_in_channel = mlp_dim[3]
            self.kernel_backbone = CropLanUnet(
                mlp_dim=(lan_emb_dim, 256, 128, 64),
                obs_in_channel=6,
                lan_in_channel=lan_in_channel,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)
        elif kernel_name == "unetl-unetc-add":
            self.kernel_backbone = CropUnetLanUNetADD(
                mlp_dim=(lan_emb_dim, 256, 128),
                obs_in_channel=6,
                lan_in_channel=1,
                out_channel=out_channel,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)
        elif kernel_name == "unetl-unetc-cat":
            self.kernel_backbone = CropUnetLanUNetCAT(
                mlp_dim=(lan_emb_dim, 256, 128),
                obs_in_channel=6,
                lan_in_channel=1,
                lan_out_channel=3,
                crop_out_channel=3,
                n_middle_channels=(16, 32, 64, 128),
                kernel_size=3,
            ).to(device)
        elif kernel_name == "eunet":
            self.kernel_backbone = EquiUnet(
                in_dim=6,
                out_dim=out_channel,
                N=4,
                middle_dim=(16, 32, 64, 128),
                init=init,
            ).to(device)
        else:
            NotImplementedError

        self.transitor = Transitor(
            device,
            n_rotations=180,
            lmax=36,
            quotient=self.quotient,
            conditioned=True,
            c=72,
        )
        self.parameter = list(self.obs_net.parameters()) + list(
            self.kernel_backbone.parameters()
        )
        self.optim = torch.optim.Adam(self.parameter, lr=1e-4)
        print(
            "phi", sum(p.numel() for p in self.obs_net.parameters() if p.requires_grad)
        )
        print(
            "psi",
            sum(
                p.numel() for p in self.kernel_backbone.parameters() if p.requires_grad
            ),
        )

    def encode_text(self, x):
        self.vlm_model.eval()
        with torch.no_grad():
            tokens = tokenize([x]).to(self.device)
            text_feat, text_emb = self.vlm_model.encode_text_with_embeddings(tokens)
        # text_mask = torch.where(tokens.float()==0., tokens.float(), 1.)  # [1, max_token_len]
        text_mask = None
        return text_feat, text_emb, text_mask

    def forward(
        self,
        in_img,
        inp_clip_features,
        lan,
        p,
        subtask,
        softmax=True,
        train=True,
        crop_source=None,
    ):
        input_data = np.pad(in_img, self.padding, mode="constant")
        if self.parse:
            input_clip_tensor = inp_clip_features[..., 2:3]
        else:
            input_clip_tensor = inp_clip_features[..., 0:1]
        input_clip_tensor = np.pad(
            input_clip_tensor, self.padding, mode="constant"
        ).transpose(
            2, 0, 1
        )  # place clip feature
        input_clip_tensor = (
            torch.from_numpy(input_clip_tensor)
            .to(torch.float)
            .unsqueeze(0)
            .to(self.device)
        )
        # if self.preprocess is not None:
        #     input_data = self.preprocess(img_unprocessed)
        # else:
        #     raise RuntimeError('img preprocess not found in pick_network.py')
        #     #input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape).transpose(0, 3, 1, 2)
        input_tensor = torch.from_numpy(input_data).to(torch.float).to(self.device)
        # get language_emd
        if self.parse:
            pick_lan, lan = parse_instruction(self.task, lan, subtask)
        lan_emd, _, _ = self.encode_text(lan)
        lan_emd = lan_emd.to(torch.float).to(self.device)

        # The crop
        if crop_source is None:
            crop_source = in_img
        crop = np.pad(crop_source, self.padding_1, mode="constant")
        crop = self.preprocess(crop)
        in_shape = (1,) + crop.shape
        crop = crop.reshape(in_shape).transpose(0, 3, 1, 2)
        # pivot = np.array([p[1], p[0]]) + self.pad_size_1 # the pivot in the entrire image with 96/2 padding each side
        crop = crop[
            :, :, p[0] : (p[0] + self.crop_size_1), p[1] : (p[1] + self.crop_size_1)
        ]
        crop_input = torch.from_numpy(crop).float().to(self.device)
        if not train:
            self.obs_net.eval()
            self.kernel_backbone.eval()
            with torch.no_grad():
                if (
                    self.model_name == "unetl"
                    or self.model_name == "unetln"
                    or self.model_name == "pre_unetl"
                ):
                    logits = self.obs_net(input_tensor, lan_emd)
                elif self.model_name in ["unetl-clip-add", "cliport-similarity-head"]:
                    logits = self.obs_net(input_tensor, lan, self.vlm_model)
                elif self.model_name in [
                    "cliport-lat",
                    "cliport-similarity-head-hugging",
                ]:
                    logits = self.obs_net(input_tensor, lan)
                elif self.model_name in ["cliport-lat-score-vit-postLinearMul"]:
                    logits = self.obs_net(
                        input_tensor, input_clip_tensor, lan, dist=self.dist
                    )
                elif self.model_name in [
                    "unetl-score-vit-postAdd",
                    "unetl-score-vit-postMul",
                    "unetl-score-vit-postLinearAdd",
                    "unetl-score-vit-postCat",
                    "eunet-score-vit-postLinearAdd",
                    "unetl-score-vit-postLinearMul",
                    "unetl-score-vit-postLinearMulSoft",
                    "unetl-score-vit-postLinearMul-m",
                    "unetl-score-vit-postLinearMulSigmoid",
                    "unetlSig-score-vit-postLinearAddSig",
                    "unetl-score-vit-priorLinearMul",
                    "unet-score-vit-postLinearMul",
                ]:
                    logits = self.obs_net(
                        input_tensor, input_clip_tensor, lan_emd, self.dist
                    )
                else:
                    logits = self.obs_net(input_tensor)

                if self.kernel_name in [
                    "unetl-unetc-add",
                    "unetl-unetc-cat",
                    "unet-lc",
                ]:
                    kernel = self.kernel_backbone(crop_input, lan_emd)
                else:
                    kernel = self.kernel_backbone(crop_input)  # default unet

        else:
            # the train mode is enabled in self.train function
            if (
                self.model_name == "unetl"
                or self.model_name == "unetln"
                or self.model_name == "pre_unetl"
            ):
                logits = self.obs_net(input_tensor, lan_emd)
            elif self.model_name in ["unetl-clip-add", "cliport-similarity-head"]:
                logits = self.obs_net(input_tensor, lan, self.vlm_model)
            elif self.model_name in ["cliport-lat", "cliport-similarity-head-hugging"]:
                logits = self.obs_net(input_tensor, lan)
            elif self.model_name in ["cliport-lat-score-vit-postLinearMul"]:
                logits = self.obs_net(
                    input_tensor, input_clip_tensor, lan, dist=self.dist
                )
            elif self.model_name in [
                "unetl-score-vit-postAdd",
                "unetl-score-vit-postMul",
                "unetl-score-vit-postLinearAdd",
                "unetl-score-vit-postCat",
                "eunet-score-vit-postLinearAdd",
                "unetl-score-vit-postLinearMul",
                "unetl-score-vit-postLinearMulSoft",
                "unetl-score-vit-postLinearMul-m",
                "unetl-score-vit-postLinearMulSigmoid",
                "unetlSig-score-vit-postLinearAddSig",
                "unetl-score-vit-priorLinearMul",
                "unet-score-vit-postLinearMul",
            ]:
                logits = self.obs_net(
                    input_tensor, input_clip_tensor, lan_emd, self.dist
                )
            else:
                logits = self.obs_net(input_tensor)

            if self.kernel_name in [
                "unetl-unetc-add",
                "unetl-unetc-cat",
                "unet-lc",
            ]:
                kernel = self.kernel_backbone(crop_input, lan_emd)
            else:
                kernel = self.kernel_backbone(crop_input)  # default unet

        if self.enable_steer:
            kernel = self.transitor.to_fourier_kernel(kernel, plot=False)
            output = self.transitor.ast(kernel, logits)
            output = self.transitor.to_spatial(output)
        else:
            output = F.conv2d(input=logits, weight=kernel)
            output = output[
                ..., : inp_clip_features.shape[0], : inp_clip_features.shape[1]
            ]

        # plot_imgs(in_img,logits,output)
        # print(logits.shape)
        # print(kernel.shape)
        # print(output.shape)
        if softmax:
            output_shape = output.shape
            output = output.reshape(-1)
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape[1:]).detach().cpu().numpy()
            output = output.transpose(1, 2, 0)
            # print(output)

        return output

    def train_step(
        self, in_img, inp_clip_features, lan, p, q, theta, subtask, backprop=True
    ):
        """Transport pixel p to pixel q.

        Args:
          in_img: input image.
          p: pixel (y, x)
          q: pixel (y, x)
          theta: rotation label in radians.
          backprop: True if backpropagating gradients.

        Returns:
          loss: training loss.
        """
        # print('hello from equ_transporter')
        self.obs_net.train()
        self.kernel_backbone.train()
        in_img = in_img.copy()
        output = self.forward(in_img, inp_clip_features, lan, p, subtask, softmax=False)
        output = output.reshape(1, -1)

        itheta = theta / (2 * np.pi / self.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.n_rotations
        # Get one-hot pixel label map.
        label_size = (self.n_rotations,) + in_img.shape[:2]
        label = torch.zeros(label_size, dtype=torch.long, device=self.device)
        label[
            itheta,
            q[0],
            q[1],
        ] = 1
        label = label.reshape(1, -1)
        label = torch.argmax(label).unsqueeze(dim=0)
        # Get loss
        loss = F.cross_entropy(input=output, target=label)

        if backprop:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        self.iters += 1
        return np.float32(loss.item())
