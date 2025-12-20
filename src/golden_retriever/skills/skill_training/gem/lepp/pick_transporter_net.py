import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import torch
import torch.nn.functional as F
from lepp.clip_revised.clip import tokenize
from lepp.parser import parse_instruction
from lepp.pick_net import Attention as PickAttention
from lifter import Transitor


class Attention(PickAttention):
    def __init__(
        self,
        cfg,
        device,
        preprocess=None,
        init=False,
        model_name="unetl",
        kernel_name="unetl",
        vlm_name="clip",
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
            lan_kernel,
            dist,
            vlm_model,
            lan_emb_dim,
        )

        self.model_name = model_name
        in_shape = (320, 160, 6)

        if cfg["lepp"]["linear_fuser"]:
            self.linear_fuser = torch.nn.Conv2d(
                self.out_channel, self.out_channel, kernel_size=65, padding=32
            ).to(self.device)
        else:
            self.linear_fuser = torch.nn.Identity().to(self.device)

        self.transitor = Transitor(
            device,
            n_rotations=180,
            lmax=36,
            quotient=self.quotient,
            conditioned=True,
            c=72,
        )
        self.parameter = list(self.obs_net.parameters())
        self.optim = torch.optim.Adam(self.parameter, lr=1e-4)
        print(
            "phi", sum(p.numel() for p in self.obs_net.parameters() if p.requires_grad)
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
        self, in_img, inp_clip_features, lan, subtask, softmax=True, train=True
    ):
        input_data = in_img
        # TODO: need to add a argument clip_type=="separate" or "entire". Now, it is separate
        if self.parse:
            input_clip_tensor = inp_clip_features[..., 1:2]
        else:
            input_clip_tensor = inp_clip_features[..., 0:1]
        input_clip_tensor = input_clip_tensor.transpose(2, 0, 1)  # pick clip feature
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
            lan, place_lan = parse_instruction(self.task, lan, subtask)
        lan_emd, _, _ = self.encode_text(lan)
        lan_emd = lan_emd.to(torch.float).to(self.device)
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
                    "unet-score-vit-postLinearMul",
                ]:
                    logits = self.obs_net(
                        input_tensor, input_clip_tensor, lan, dist=self.dist
                    )

                else:
                    logits = self.obs_net(input_tensor)

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
                "unet-score-vit-postLinearMul",
            ]:
                logits = self.obs_net(
                    input_tensor, input_clip_tensor, lan_emd, dist=self.dist
                )
            else:
                logits = self.obs_net(input_tensor)

        output = self.linear_fuser(logits)

        # c0 = self.padding[:2, 0]
        # c1 = c0 + in_img.shape[:2]
        # output = output[:, :, c0[0]:c1[0], c0[1]:c1[1]]

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
