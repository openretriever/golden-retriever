## the input is 1 x 512
# MLP : 512 -> 256 -> 128
# Unet-COV2D: 128 X 128 -> 64 X 64 --> 32 X 32 -> 16 X 16 --> 32 X 32 -> 64 X 64

from collections import OrderedDict

import torch
import torch.nn as nn


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
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
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
                [("dec-res-8", ResBlock(2 * self.l4_c, self.l3_c, kernel_size))]
            )
        )

        self.conv_up_4 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-4", ResBlock(2 * self.l3_c, self.l2_c, kernel_size))]
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
        self, feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16
    ):
        concat_8 = torch.cat((feature_map_8, self.upsample_16_8(feature_map_16)), dim=1)
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat(
            (feature_map_4, self.upsample_8_4(feature_map_up_8)), dim=1
        )
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

    def forward(self, x):
        x = self.preprocess(x, dist="transporter")
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16
        )
        return out


# x = torch.rand(1,1,128,128).cuda()
# unet = ResUnet(in_channel=1,out_channel=3).cuda()
# y = unet(x)
# print(y.shape)
