## the input is 1 x 512
# MLP : 512 -> 256 -> 128
# Unet-COV2D: 128 X 128 -> 64 X 64 --> 32 X 32 -> 16 X 16 --> 32 X 32 -> 64 X 64

from collections import OrderedDict

import torch
import torch.nn as nn
from lepp.clip_revised.clip import preprocess as clip_preprocess
from lepp.clip_revised.clip import tokenize
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    CLIPTextModelWithProjection,
)


class ResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, last_relu=True):
        super(ResBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=1,
            )
        )
        self.relu = nn.ReLU(inplace=True)
        self.last_relu = last_relu
        self.scale = None
        if input_channels != hidden_dim:
            self.scale = nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    stride=1,
                )
            )

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)

        if self.scale:
            out += self.scale(residual)
        else:
            out += residual

        if self.last_relu:
            out = self.relu(out)
        return out


class ResUnet(torch.nn.Module):
    def __init__(
        self,
        preprocess,
        lan_emb_dim=1024,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
    ):
        super().__init__()
        self.preprocess = preprocess

        assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]

        self.conv_down_1 = torch.nn.Sequential(
            OrderedDict([("enc-res-1", ResBlock(in_channel, self.l1_c, kernel_size))])
        )
        self.conv_down_2 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-2", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-2", ResBlock(self.l1_c, self.l2_c, kernel_size)),
                ]
            )
        )
        self.conv_down_4 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-4", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-4", ResBlock(self.l2_c, self.l3_c, kernel_size)),
                ]
            )
        )

        self.conv_down_8 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-8", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-8", ResBlock(self.l3_c, self.l4_c, kernel_size)),
                ]
            )
        )
        self.conv_down_16 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-16", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-16", ResBlock(self.l4_c, self.l4_c, kernel_size)),
                ]
            )
        )

        self.conv_up_8 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-8", ResBlock(4 * self.l4_c, self.l3_c, kernel_size))]
            )
        )

        self.conv_up_4 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-4", ResBlock(4 * self.l3_c, self.l2_c, kernel_size))]
            )
        )

        self.conv_up_2 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-2", ResBlock(2 * self.l2_c, self.l1_c, kernel_size))]
            )
        )

        self.conv_up_1 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-2", ResBlock(2 * self.l1_c, self.l1_c, kernel_size))]
            )
        )

        self.final = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "dec-final",
                        torch.nn.Conv2d(
                            in_channels=self.l1_c,
                            out_channels=out_channel,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                        ),
                    )
                ]
            )
        )

        self.upsample_16_8 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_8_4 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_4_2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_2_1 = nn.Upsample(scale_factor=2, mode="bilinear")

        # lan_projection
        self.lan_ln1 = nn.Linear(lan_emb_dim, n_middle_channels[-1] * 2)
        self.lan_ln2 = nn.Linear(lan_emb_dim, n_middle_channels[-2] * 2)

    def _init(self):
        pass

    def forwardEncoder(self, x):
        feature_map_1 = self.conv_down_1(x)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        return (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        )

    def forwardDecoder(
        self,
        feature_map_1,
        feature_map_2,
        feature_map_4,
        feature_map_8,
        feature_map_16,
        lan_emd,
    ):
        concat_8 = torch.cat((feature_map_8, self.upsample_16_8(feature_map_16)), dim=1)
        lan8 = (
            self.lan_ln1(lan_emd)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1)
            .repeat(1, 1, concat_8.shape[-2], concat_8.shape[-1])
        )
        concat_8 = torch.cat((concat_8, lan8), dim=1)
        # print('add lan emd to unet bottleneck')
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat(
            (feature_map_4, self.upsample_8_4(feature_map_up_8)), dim=1
        )
        lan4 = (
            self.lan_ln2(lan_emd)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1)
            .repeat(1, 1, concat_4.shape[-2], concat_4.shape[-1])
        )
        concat_4 = torch.cat((concat_4, lan4), dim=1)
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat(
            (feature_map_2, self.upsample_4_2(feature_map_up_4)), dim=1
        )
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat(
            (feature_map_1, self.upsample_2_1(feature_map_up_2)), dim=1
        )
        feature_map_up_1 = self.conv_up_1(concat_1)

        feature_map_final = self.final(feature_map_up_1)

        return feature_map_final

    def forward(self, x, lan_emd=None):
        x = self.preprocess(x, dist="transporter")
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )
        return out


class ResUnetCLIPPostADD(ResUnet):
    def __init__(
        self,
        preprocess,
        lan_emb_dim=1024,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
    ):
        super().__init__(
            preprocess,
            lan_emb_dim,
            in_channel,
            out_channel,
            n_middle_channels,
            kernel_size,
            lan_emd,
        )

    def forward(self, x, clip_feature, lan_emd=None):
        x = self.preprocess(x, dist="transporter")
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )
        out += clip_feature
        return out


class ResUnetCLIPPostMul(ResUnet):
    def __init__(
        self,
        preprocess,
        lan_emb_dim=1024,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
    ):
        super().__init__(
            preprocess,
            lan_emb_dim,
            in_channel,
            out_channel,
            n_middle_channels,
            kernel_size,
            lan_emd,
        )
        self.conv_clip = nn.Conv2d(1, out_channel, kernel_size=9, padding=4)

    def forward(self, x, clip_feature, lan_emd=None, dist="transporter"):
        x = self.preprocess(x, dist=dist)
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )
        # clip_feature[clip_feature<0.5] = 0.5 # heuristically prevent inaccurate clip map destroy some info
        clip_feature = self.conv_clip(clip_feature)
        out *= clip_feature
        return out


class ResUnetCLIPPostLinearADD(ResUnet):
    def __init__(
        self,
        preprocess,
        lan_emb_dim=1024,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
    ):
        super().__init__(
            preprocess,
            lan_emb_dim,
            in_channel,
            out_channel,
            n_middle_channels,
            kernel_size,
            lan_emd,
        )
        shape_out = 3
        self.shape_conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.shape_relu1 = nn.ReLU()
        self.shape_conv2 = nn.Conv2d(3, shape_out, kernel_size=3, padding=1)

        self.conv_clip = nn.Conv2d(1 + shape_out, out_channel, kernel_size=9, padding=4)

    def forward(self, x, clip_feature, lan_emd=None, dist="transporter"):
        x = self.preprocess(x, dist=dist)
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )
        device = x.device
        depth = x[:1, -1:, ...].detach().clone().to(device)
        shape_feature = self.shape_conv2(self.shape_relu1(self.shape_conv1(depth)))
        clip_feature = torch.cat(
            [clip_feature, shape_feature], axis=1
        )  # encode some shape info
        clip_feature = self.conv_clip(clip_feature)
        out += clip_feature
        return out


class ResUnetSigCLIPPostLinearADDSig(ResUnet):
    def __init__(
        self,
        preprocess,
        lan_emb_dim=1024,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
    ):
        super().__init__(
            preprocess,
            lan_emb_dim,
            in_channel,
            out_channel,
            n_middle_channels,
            kernel_size,
            lan_emd,
        )
        shape_out = 3
        self.shape_conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.shape_relu1 = nn.ReLU()
        self.shape_conv2 = nn.Conv2d(3, shape_out, kernel_size=3, padding=1)

        self.conv_clip = nn.Conv2d(1 + shape_out, out_channel, kernel_size=9, padding=4)
        # self.conv_out = nn.Conv2d(out_channel, out_channel, kernel_size=9, padding=4)
        self.signoid_unet = nn.Sigmoid()
        self.signoid_clip = nn.Sigmoid()

    def forward(self, x, clip_feature, lan_emd=None, dist="transporter"):
        x = self.preprocess(x, dist=dist)
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )
        device = x.device
        depth = x[:1, -1:, ...].detach().clone().to(device)
        shape_feature = self.shape_conv2(self.shape_relu1(self.shape_conv1(depth)))
        clip_feature = torch.cat(
            [clip_feature, shape_feature], axis=1
        )  # encode some shape info
        clip_feature = self.conv_clip(clip_feature)
        out = self.signoid_unet(out) + self.signoid_clip(clip_feature)
        # out = self.conv_out(out)
        return out


class ResUnetCLIPPostLinearMul(ResUnet):
    def __init__(
        self,
        preprocess,
        lan_emb_dim=1024,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
    ):
        super().__init__(
            preprocess,
            lan_emb_dim,
            in_channel,
            out_channel,
            n_middle_channels,
            kernel_size,
            lan_emd,
        )
        shape_out = 3
        self.shape_conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.shape_relu1 = nn.ReLU()
        self.shape_conv2 = nn.Conv2d(3, shape_out, kernel_size=3, padding=1)

        self.conv_clip = nn.Conv2d(1 + shape_out, out_channel, kernel_size=9, padding=4)

    def forward(self, x, clip_feature, lan_emd=None, dist="transporter"):
        x = self.preprocess(x, dist=dist)
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )
        device = x.device
        depth = x[:1, -1:, ...].detach().clone().to(device)
        shape_feature = self.shape_conv2(self.shape_relu1(self.shape_conv1(depth)))
        clip_feature = torch.cat(
            [clip_feature, shape_feature], axis=1
        )  # encode some shape info
        clip_feature = self.conv_clip(clip_feature)
        out *= clip_feature
        return out


class ResUnetWoLanCLIPPostLinearMul(ResUnet):
    def __init__(
        self,
        preprocess,
        lan_emb_dim=1024,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
    ):
        super().__init__(
            preprocess,
            lan_emb_dim,
            in_channel,
            out_channel,
            n_middle_channels,
            kernel_size,
            lan_emd,
        )
        shape_out = 3
        self.shape_conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.shape_relu1 = nn.ReLU()
        self.shape_conv2 = nn.Conv2d(3, shape_out, kernel_size=3, padding=1)

        self.conv_clip = nn.Conv2d(1 + shape_out, out_channel, kernel_size=9, padding=4)

    def forward(self, x, clip_feature, lan_emd=None, dist="transporter"):
        x = self.preprocess(x, dist=dist)
        device = x.device
        lan_emd = torch.ones_like(lan_emd).to(device)
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )

        depth = x[:1, -1:, ...].detach().clone().to(device)
        shape_feature = self.shape_conv2(self.shape_relu1(self.shape_conv1(depth)))
        clip_feature = torch.cat(
            [clip_feature, shape_feature], axis=1
        )  # encode some shape info
        clip_feature = self.conv_clip(clip_feature)
        out *= clip_feature
        return out


class ResUnetCLIPPriorLinearMul(ResUnet):
    def __init__(
        self,
        preprocess,
        lan_emb_dim=1024,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
    ):
        super().__init__(
            preprocess,
            lan_emb_dim,
            in_channel,
            out_channel,
            n_middle_channels,
            kernel_size,
            lan_emd,
        )
        shape_out = 3
        self.shape_conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.shape_relu1 = nn.ReLU()
        self.shape_conv2 = nn.Conv2d(3, shape_out, kernel_size=3, padding=1)

        self.conv_clip = nn.Conv2d(1 + shape_out, 6, kernel_size=9, padding=4)

    def forward(self, x, clip_feature, lan_emd=None, dist="transporter"):
        x = self.preprocess(x, dist=dist)
        device = x.device
        depth = x[:1, -1:, ...].detach().clone().to(device)
        shape_feature = self.shape_conv2(self.shape_relu1(self.shape_conv1(depth)))
        clip_feature = torch.cat(
            [clip_feature, shape_feature], axis=1
        )  # encode some shape info
        clip_feature = self.conv_clip(clip_feature)
        x *= clip_feature

        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )

        return out


class ResUnetCLIPPostLinearMulSigmoid(ResUnet):
    def __init__(
        self,
        preprocess,
        lan_emb_dim=1024,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
    ):
        super().__init__(
            preprocess,
            lan_emb_dim,
            in_channel,
            out_channel,
            n_middle_channels,
            kernel_size,
            lan_emd,
        )
        shape_out = 3
        self.shape_conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.shape_relu1 = nn.ReLU()
        self.shape_conv2 = nn.Conv2d(3, shape_out, kernel_size=3, padding=1)

        self.conv_clip = nn.Conv2d(1 + shape_out, out_channel, kernel_size=9, padding=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, clip_feature, lan_emd=None, dist="transporter"):
        x = self.preprocess(x, dist=dist)
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )
        device = x.device
        depth = x[:1, -1:, ...].detach().clone().to(device)
        shape_feature = self.shape_conv2(self.shape_relu1(self.shape_conv1(depth)))
        clip_feature = torch.cat(
            [clip_feature, shape_feature], axis=1
        )  # encode some shape info
        clip_feature = self.conv_clip(clip_feature)
        clip_feature = self.sigmoid(clip_feature)
        out *= clip_feature
        return out


class ResUnetCLIPPostLinearMulSoft(ResUnet):
    def __init__(
        self,
        preprocess,
        lan_emb_dim=1024,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
    ):
        super().__init__(
            preprocess,
            lan_emb_dim,
            in_channel,
            out_channel,
            n_middle_channels,
            kernel_size,
            lan_emd,
        )
        shape_out = 3
        self.shape_conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.shape_relu1 = nn.ReLU()
        self.shape_conv2 = nn.Conv2d(3, shape_out, kernel_size=3, padding=1)

        self.conv_clip = nn.Conv2d(1 + shape_out, out_channel, kernel_size=9, padding=4)

    def forward(self, x, clip_feature, lan_emd=None, dist="transporter"):
        x = self.preprocess(x, dist=dist)
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )
        device = x.device
        depth = x[:1, -1:, ...].detach().clone().to(device)
        shape_feature = self.shape_conv2(self.shape_relu1(self.shape_conv1(depth)))
        clip_feature[clip_feature < 0.5] = 0.5
        clip_feature = torch.cat(
            [clip_feature, shape_feature], axis=1
        )  # encode some shape info
        clip_feature = self.conv_clip(clip_feature)
        out *= clip_feature
        return out


class ResUnetCLIPPostCat(ResUnet):
    def __init__(
        self,
        preprocess,
        lan_emb_dim=1024,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
    ):
        super().__init__(
            preprocess,
            lan_emb_dim,
            in_channel,
            out_channel,
            n_middle_channels,
            kernel_size,
            lan_emd,
        )

    def forward(self, x, clip_feature, lan_emd=None):
        x = self.preprocess(x, dist="transporter")
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )
        out = torch.cat([out, clip_feature], axis=1)
        return out


class ResUnetCLIPHeadHuggingFace(torch.nn.Module):
    def __init__(
        self,
        device,
        preprocess,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
        model_id="openai/clip-vit-base-patch32",
        lan_dim=1024,
    ):
        super().__init__()
        self.preprocess = preprocess

        assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]

        self.conv_down_1 = torch.nn.Sequential(
            OrderedDict(
                [("enc-res-1", ResBlock(in_channel + 1, self.l1_c, kernel_size))]
            )
        )
        self.conv_down_2 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-2", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-2", ResBlock(self.l1_c, self.l2_c, kernel_size)),
                ]
            )
        )
        self.conv_down_4 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-4", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-4", ResBlock(self.l2_c, self.l3_c, kernel_size)),
                ]
            )
        )

        self.conv_down_8 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-8", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-8", ResBlock(self.l3_c, self.l4_c, kernel_size)),
                ]
            )
        )
        self.conv_down_16 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-16", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-16", ResBlock(self.l4_c, self.l4_c, kernel_size)),
                ]
            )
        )

        self.conv_up_8 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-8", ResBlock(4 * self.l4_c, self.l3_c, kernel_size))]
            )
        )

        self.conv_up_4 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-4", ResBlock(4 * self.l3_c, self.l2_c, kernel_size))]
            )
        )

        self.conv_up_2 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-2", ResBlock(2 * self.l2_c, self.l1_c, kernel_size))]
            )
        )

        self.conv_up_1 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-2", ResBlock(2 * self.l1_c, self.l1_c, kernel_size))]
            )
        )

        self.final = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "dec-final",
                        torch.nn.Conv2d(
                            in_channels=self.l1_c,
                            out_channels=out_channel,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                        ),
                    )
                ]
            )
        )

        self.upsample_16_8 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_8_4 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_4_2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_2_1 = nn.Upsample(scale_factor=2, mode="bilinear")

        # lan_projection
        self.lan_ln1 = nn.Linear(lan_dim, n_middle_channels[-1] * 2)
        self.lan_ln2 = nn.Linear(lan_dim, n_middle_channels[-2] * 2)

        # set clip
        print("only support hugging's vit model")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tmodel = CLIPTextModelWithProjection.from_pretrained(model_id).to(device)

    def _init(self):
        pass

    def forwardCLIPPatch(self, x, lan_goal):
        with torch.no_grad():
            device = x.device
            rgb = x.clone()[:, :3, ...]
            kernel_size, stride = 20, 20
            patches = rgb.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
            patches = patches.contiguous().view(
                patches.shape[0] * patches.shape[1], -1, kernel_size, kernel_size
            )
            patches = patches.permute((1, 0, 2, 3))

            processor = clip_preprocess(224)
            patches = processor(patches / 255.0)

            text = self.tokenizer([lan_goal], padding=True, return_tensors="pt").to(
                device
            )
            text_features = self.tmodel(**text).text_embeds

            # patches = self.processor(
            #     images=patches,  # big patch image sent to CLIP
            #     return_tensors="pt",  # tell CLIP to return pytorch tensor
            # ).to(device).pixel_values  # too slow

            # score = self.model(**inputs)
            score = self.model(input_ids=text.input_ids, pixel_values=patches)
            scores = score.logits_per_image
            scores = scores.reshape(x.shape[0], patches.shape[0])
            # clip the scores
            scores = torch.clip(scores - scores.mean(dim=-1), 0, torch.inf)

            # normalize scores
            scores = (scores - scores.min(dim=-1).values) / (
                scores.max(dim=-1).values - scores.min(dim=-1).values
            )

            clip_feature = scores.reshape(
                x.shape[0], 1, rgb.shape[2] // kernel_size, rgb.shape[3] // kernel_size
            )
            clip_feature = nn.functional.interpolate(
                clip_feature,
                size=[x.shape[2], x.shape[3]],
                mode="bilinear",
                align_corners=True,
            )

            # Upsampler = nn.Upsample(scale_factor=stride, mode='bilinear')
        return clip_feature, text_features

    def forwardEncoder(self, x):
        feature_map_1 = self.conv_down_1(x)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        return (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        )

    def forwardDecoder(
        self,
        feature_map_1,
        feature_map_2,
        feature_map_4,
        feature_map_8,
        feature_map_16,
        lan_emd,
    ):
        concat_8 = torch.cat((feature_map_8, self.upsample_16_8(feature_map_16)), dim=1)
        lan8 = (
            self.lan_ln1(lan_emd)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1)
            .repeat(1, 1, concat_8.shape[-2], concat_8.shape[-1])
        )
        concat_8 = torch.cat((concat_8, lan8), dim=1)
        # print('add lan emd to unet bottleneck')
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat(
            (feature_map_4, self.upsample_8_4(feature_map_up_8)), dim=1
        )
        lan4 = (
            self.lan_ln2(lan_emd)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1)
            .repeat(1, 1, concat_4.shape[-2], concat_4.shape[-1])
        )
        concat_4 = torch.cat((concat_4, lan4), dim=1)
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat(
            (feature_map_2, self.upsample_4_2(feature_map_up_4)), dim=1
        )
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat(
            (feature_map_1, self.upsample_2_1(feature_map_up_2)), dim=1
        )
        feature_map_up_1 = self.conv_up_1(concat_1)

        feature_map_final = self.final(feature_map_up_1)

        return feature_map_final

    def forward(self, x, lan_goal):
        clip_feature, lan_emd = self.forwardCLIPPatch(x, lan_goal)
        x = self.preprocess(x, dist="transporter")
        x = torch.cat((x, clip_feature), dim=1)
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )
        return out


class ResUnetCLIPHead(torch.nn.Module):
    def __init__(
        self,
        preprocess,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=False,
        lan_dim=1024,
    ):
        super().__init__()
        self.preprocess = preprocess

        assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]

        self.conv_down_1 = torch.nn.Sequential(
            OrderedDict(
                [("enc-res-1", ResBlock(in_channel + 1, self.l1_c, kernel_size))]
            )
        )
        self.conv_down_2 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-2", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-2", ResBlock(self.l1_c, self.l2_c, kernel_size)),
                ]
            )
        )
        self.conv_down_4 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-4", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-4", ResBlock(self.l2_c, self.l3_c, kernel_size)),
                ]
            )
        )

        self.conv_down_8 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-8", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-8", ResBlock(self.l3_c, self.l4_c, kernel_size)),
                ]
            )
        )
        self.conv_down_16 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-16", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-16", ResBlock(self.l4_c, self.l4_c, kernel_size)),
                ]
            )
        )

        self.conv_up_8 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-8", ResBlock(4 * self.l4_c, self.l3_c, kernel_size))]
            )
        )

        self.conv_up_4 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-4", ResBlock(4 * self.l3_c, self.l2_c, kernel_size))]
            )
        )

        self.conv_up_2 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-2", ResBlock(2 * self.l2_c, self.l1_c, kernel_size))]
            )
        )

        self.conv_up_1 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-2", ResBlock(2 * self.l1_c, self.l1_c, kernel_size))]
            )
        )

        self.final = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "dec-final",
                        torch.nn.Conv2d(
                            in_channels=self.l1_c,
                            out_channels=out_channel,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                        ),
                    )
                ]
            )
        )

        self.upsample_16_8 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_8_4 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_4_2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_2_1 = nn.Upsample(scale_factor=2, mode="bilinear")

        # lan_projection
        self.lan_ln1 = nn.Linear(lan_dim, n_middle_channels[-1] * 2)
        self.lan_ln2 = nn.Linear(lan_dim, n_middle_channels[-2] * 2)

    def _init(self):
        pass

    def forwardCLIPPatch(self, x, lan_goal, vlm_model):
        with torch.no_grad():
            device = x.device
            rgb = x.clone()[:, :3, ...]

            kernel_size, stride = 20, 20
            patches = rgb.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
            patches = patches.contiguous().view(
                patches.shape[0] * patches.shape[1], -1, kernel_size, kernel_size
            )
            patches = patches.permute((1, 0, 2, 3)) / 255

            processor = clip_preprocess(vlm_model.visual.input_resolution)
            patches = processor(patches)

            text = tokenize([lan_goal]).to(device)
            text_features = vlm_model.encode_text(text).float()

            logits_per_image, logits_per_text = vlm_model(patches, text)
            scores = logits_per_image.reshape(x.shape[0], patches.shape[0])
            # clip the scores
            scores = torch.clip(scores - scores.mean(dim=-1), 0, torch.inf)

            # normalize scores
            scores = (scores - scores.min(dim=-1).values) / (
                scores.max(dim=-1).values - scores.min(dim=-1).values
            )

            clip_feature = scores.reshape(
                x.shape[0], 1, rgb.shape[2] // kernel_size, rgb.shape[3] // kernel_size
            )

            clip_feature = nn.functional.interpolate(
                clip_feature,
                size=[x.shape[2], x.shape[3]],
                mode="bilinear",
                align_corners=True,
            ).float()
            # clip_feature[(x[:,:3].sum(dim=1)==0.).unsqueeze(1)] = 0.  # filter out black areas

            # Upsampler = nn.Upsample(scale_factor=stride, mode='bilinear')
        return clip_feature, text_features

    def forwardEncoder(self, x):
        feature_map_1 = self.conv_down_1(x)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        return (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        )

    def forwardDecoder(
        self,
        feature_map_1,
        feature_map_2,
        feature_map_4,
        feature_map_8,
        feature_map_16,
        lan_emd,
    ):
        concat_8 = torch.cat((feature_map_8, self.upsample_16_8(feature_map_16)), dim=1)
        lan8 = (
            self.lan_ln1(lan_emd)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1)
            .repeat(1, 1, concat_8.shape[-2], concat_8.shape[-1])
        )
        concat_8 = torch.cat((concat_8, lan8), dim=1)
        # print('add lan emd to unet bottleneck')
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat(
            (feature_map_4, self.upsample_8_4(feature_map_up_8)), dim=1
        )
        lan4 = (
            self.lan_ln2(lan_emd)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1)
            .repeat(1, 1, concat_4.shape[-2], concat_4.shape[-1])
        )
        concat_4 = torch.cat((concat_4, lan4), dim=1)
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat(
            (feature_map_2, self.upsample_4_2(feature_map_up_4)), dim=1
        )
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat(
            (feature_map_1, self.upsample_2_1(feature_map_up_2)), dim=1
        )
        feature_map_up_1 = self.conv_up_1(concat_1)

        feature_map_final = self.final(feature_map_up_1)

        return feature_map_final

    def forward(self, x, lan_goal, vlm_model):
        clip_feature, lan_emd = self.forwardCLIPPatch(x, lan_goal, vlm_model)
        x = self.preprocess(x, dist="transporter")
        x = torch.cat((x, clip_feature), dim=1)
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            lan_emd,
        )
        return out
