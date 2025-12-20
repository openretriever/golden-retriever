import os
import sys

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
from lepp.equ_unet import EquiUnetCLIPPostLinearADD
from lepp.kernel_backbone_mlp_conv import BackBone, BackBoneDiffusion, BackBoneM
from lepp.parser import parse_instruction
from lepp.pretrain_kernel_backbone_mlp_conv import PreBackBone
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


class Attention(torch.nn.Module):
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

        # ----------------------------------------------------------------------

        self.model_name = model_name
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

        if model_name == "unetl-m" or model_name == "pre_unetl-m":
            print("scratch unetl")
            self.obs_net = ResUnetLan(
                preprocess=utils.preprocess,
                lan_emb_dim=lan_emb_dim,
                in_channel=6,
                out_channel=out_channel,
                n_middle_channels=(32, 64, 128, 128),
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

        elif model_name == "unetl-clip-add":
            print("obsnet: scratch unetl-clip-add")
            NotImplementedError

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

            # need change
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

        # elif model_name == 'vit_unet':
        #     print('scratch vit_unet')
        #     self.obs_net = VitUnet(in_channel=6,out_channel=out_channel).to(self.device)

        # elif model_name == 'vit_unet_hi':
        #     print('scratch vit_unet_hi')
        #     self.obs_net = VitUnetHi(in_channel=6,out_channel=out_channel).to(self.device)

        # elif model_name == 'vitconv_unet_hi':
        #     print('scratch vitconv_unet_hi')
        #     self.obs_net = VitConvUnetHi(in_channel=6,out_channel=out_channel).to(self.device)

        if "pre" in kernel_name:
            print("pre-train lan-img")
            self.kernel_backbone = PreBackBone(device, out_channel=3).to(self.device)
        else:
            if kernel_name == "unetln":
                lan_emd = False
                print("unconditioned kernel")
                self.kernel_backbone = BackBone(
                    mlp_dim=(lan_emb_dim, 256, 128),
                    in_channel=1,
                    out_channel=out_channel,
                    n_middle_channels=(16, 32, 64, 128),
                    kernel_size=3,
                    lan_emd=lan_emd,
                ).to(device)
            elif kernel_name == "unetl":
                lan_emd = True
                print("language conditioned kernel")
                self.kernel_backbone = BackBone(
                    mlp_dim=(lan_emb_dim, 256, 128),
                    in_channel=1,
                    out_channel=out_channel,
                    n_middle_channels=(16, 32, 64, 128),
                    kernel_size=3,
                    lan_emd=lan_emd,
                ).to(device)
            elif kernel_name == "unetl-m":
                lan_emd = True
                print("language conditioned kernel, middle size")
                self.kernel_backbone = BackBoneM(
                    mlp_dim=(lan_emb_dim, 256, 128, 128),
                    in_channel=1,
                    out_channel=out_channel,
                    n_middle_channels=(16, 32, 64, 128, 256, 512),
                    kernel_size=3,
                    lan_emd=lan_emd,
                ).to(device)
            elif kernel_name == "unetl-loop":
                lan_emd = True
                print("language conditioned kernel + loop")
                n_diff = 5
                self.kernel_backbone = BackBoneDiffusion(
                    mlp_dim=(lan_emb_dim, 256, 128),
                    in_channel=1,
                    out_channel=out_channel,
                    n_middle_channels=(16, 32, 64, 128),
                    kernel_size=3,
                    lan_emd=lan_emd,
                    n_diff=n_diff,
                ).to(device)
            elif kernel_name == "unetl-diff":
                # TODO
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
        self, in_img, inp_clip_features, lan, subtask, softmax=True, train=True
    ):
        input_data = np.pad(in_img, self.padding, mode="constant")
        # TODO: need to add a argument clip_type=="separate" or "entire". Now, it is separate
        if self.parse:
            input_clip_tensor = inp_clip_features[..., 1:2]
        else:
            input_clip_tensor = inp_clip_features[..., 0:1]
        input_clip_tensor = np.pad(
            input_clip_tensor, self.padding, mode="constant"
        ).transpose(
            2, 0, 1
        )  # pick clip feature
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
            lan, pick_lan = parse_instruction(self.task, lan, subtask)

        lan_emd, _, _ = self.encode_text(lan)
        lan_emd = lan_emd.to(torch.float).to(self.device)
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
                        input_tensor, input_clip_tensor, lan_emd, dist=self.dist
                    )

                else:
                    logits = self.obs_net(input_tensor)

                kernel = self.kernel_backbone(lan_emd)

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
                    input_tensor, input_clip_tensor, lan_emd, dist=self.dist
                )
            else:
                logits = self.obs_net(input_tensor)

            kernel = self.kernel_backbone(lan_emd)

        if self.enable_steer:
            kernel = self.transitor.to_fourier_kernel(kernel, plot=False)
            output = self.transitor.ast(kernel, logits)
            output = self.transitor.to_spatial(output)
        else:
            output = F.conv2d(input=logits, weight=kernel)
            output = output[
                ..., : inp_clip_features.shape[0], : inp_clip_features.shape[1]
            ]

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

    def train_step(
        self, in_img, inp_clip_features, lan_emd, p, theta, subtask, backprop=True
    ):
        self.obs_net.train()
        self.kernel_backbone.train()
        in_img = in_img.copy()
        output = self.forward(
            in_img, inp_clip_features, lan_emd, subtask, softmax=False
        )

        # setup the label
        if self.quotient:
            # [0:35]
            theta = (theta + 2 * np.pi) % (np.pi)
            theta_i = theta // (2 * np.pi / self.n_rotations)
        else:
            # [0:71]
            theta = (theta + 2 * np.pi) % (2 * np.pi)
            theta_i = theta // (2 * np.pi / self.n_rotations)
        theta_i = np.int32(np.round(theta_i))

        # get the one-hot label
        # label_size = (36,) + in_img.shape[:2] if self.quotient else (72,) + in_img.shape[:2]
        # label = torch.zeros(label_size, dtype=torch.long, device=self.device)
        # label[theta_i, p[0], p[1]] = 1
        # label = label.reshape(-1)
        # label = torch.argmax(label).unsqueeze(dim=0)

        # more each way to get the one hot label
        shape = output.shape  # b x c x h x w
        label = theta_i * (shape[-1] * shape[-2]) + p[0] * shape[-1] + p[1]
        label = torch.as_tensor(label).unsqueeze(dim=0).to(self.device)
        # print(shape)
        # print(theta_i)
        # print(p)
        # print(label,'2')
        output = output.reshape(1, -1)
        loss = F.cross_entropy(input=output, target=label)

        if backprop:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return np.float32(loss.item())

    def load(self, path1, path2):
        # safe operation for e2cnn
        self.obs_net.eval()
        self.obs_net.load_state_dict(torch.load(path1, map_location=self.device))

        self.kernel_backbone.eval()
        self.kernel_backbone.load_state_dict(
            torch.load(path2, map_location=self.device)
        )

    def save(self, path1, path2):
        # torch.cuda.empty_cache()
        # safe operation for e2cnn
        self.obs_net.eval()
        torch.save(self.obs_net.state_dict(), path1)
        self.kernel_backbone.eval()
        torch.save(self.kernel_backbone.state_dict(), path2)


# device = torch.device('cuda')
# att = Attention(device)
# img = np.random.random((320,160,6))
# lan_emd = np.random.random((1,512))
# out = att.forward(img,lan_emd,softmax=False)
# print(out.shape)
