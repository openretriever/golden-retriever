import os
import sys

import kornia
import numpy as np
import torch
import torch.nn.functional as F

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import numpy as np
import torch
import torch.nn.functional as F
from lepp.parser import parse_instruction
from lepp.place_net import Transport


class TransportWoFourier(Transport):
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
        super().__init__(
            cfg,
            device,
            preprocess,
            init,
            model_name,
            kernel_name,
            vlm_name,
            crop_size,
            lan_kernel,
            dist,
            vlm_model,
            lan_emb_dim,
        )
        self.pad_size_2 = int(self.crop_size_1 / 2)

        if cfg["lepp"]["linear_fuser"]:
            self.linear_fuser = torch.nn.Conv2d(
                self.out_channel, self.out_channel, kernel_size=65, padding=32
            ).to(self.device)
        else:
            self.linear_fuser = torch.nn.Identity().to(self.device)

    def forward(
        self, in_img, inp_clip_features, lan, p, subtask, softmax=True, train=True
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
        crop = np.pad(in_img, self.padding_1, mode="constant")
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
                if self.model_name in ["unetl", "unetln", "pre_unetl", "unetl-m"]:
                    logits = self.obs_net(input_tensor, lan_emd)
                elif self.model_name in ["unetl-clip-add", "cliport-similarity-head"]:
                    logits = self.obs_net(input_tensor, lan, self.vlm_model)
                elif self.model_name in [
                    "cliport-lat",
                    "cliport-similarity-head-hugging",
                ]:
                    logits = self.obs_net(input_tensor, lan)
                elif self.model_name in [
                    "unetl-score-vit-postAdd",
                    "unetl-score-vit-postMul",
                    "unetl-score-vit-postLinearAdd",
                    "unetl-score-vit-postCat",
                    "eunet-score-vit-postLinearAdd",
                    "unetl-score-vit-postLinearMul",
                    "unetl-score-vit-postLinearMulSoft",
                    "unetl-score-vit-postLinearMul-m",
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
            if self.model_name in ["unetl", "unetln", "pre_unetl", "unetl-m"]:
                logits = self.obs_net(input_tensor, lan_emd)
            elif self.model_name in ["unetl-clip-add", "cliport-similarity-head"]:
                logits = self.obs_net(input_tensor, lan, self.vlm_model)
            elif self.model_name in ["cliport-lat", "cliport-similarity-head-hugging"]:
                logits = self.obs_net(input_tensor, lan)
            elif self.model_name in [
                "unetl-score-vit-postAdd",
                "unetl-score-vit-postMul",
                "unetl-score-vit-postLinearAdd",
                "unetl-score-vit-postCat",
                "eunet-score-vit-postLinearAdd",
                "unetl-score-vit-postLinearMul",
                "unetl-score-vit-postLinearMulSoft",
                "unetl-score-vit-postLinearMul-m",
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

        # logits = self.linear_fuser(logits)
        # Rotate the cropped feature conterclockwise and conduct another crop to get 65x65 kernels
        pivot = int(self.crop_size_1 / 2)
        assert pivot == int(kernel.shape[-1] / 2)
        # print('pivot',pivot)
        half_length = self.pad_size_2
        l, r = pivot - half_length, pivot + half_length + 1
        b, u = pivot - half_length, pivot + half_length + 1
        kernel = kernel.repeat(self.n_rotations, 1, 1, 1)
        kernel = kornia.geometry.rotate(
            kernel,
            torch.from_numpy(
                np.linspace(
                    0.0, 360.0, self.n_rotations, endpoint=False, dtype=np.float32
                )
            ).to(self.device),
            mode="nearest",
        )
        kernel = kernel[:, :, l:r, b:u]
        output = F.conv2d(input=logits, weight=kernel)
        output = output[..., : inp_clip_features.shape[0], : inp_clip_features.shape[1]]

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
