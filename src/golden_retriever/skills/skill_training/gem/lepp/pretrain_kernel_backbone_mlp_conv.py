## the input is 1 x 512
# MLP : 512 -> 256 -> 128
# Unet-COV2D: 128 X 128 -> 64 X 64 --> 32 X 32 -> 16 X 16 --> 32 X 32 -> 64 X 64

from collections import OrderedDict

import torch
import torch.nn as nn
from pretrain_lan_kernel import PretrainedLanKernel


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


# x = torch.rand(1,1,100,100)
# resb1 = ResBlock(input_channels=1, hidden_dim=10,kernel_size=3)
# y = resb1(x)
# print(y.shape)

# x = torch.rand(1,100)
# fc = torch.nn.Linear(in_features=100,out_features=50,bias=True)
# y = fc(x)
# print(y.shape)


class PreBackBone(torch.nn.Module):
    def __init__(
        self,
        device,
        out_channel=3,
        n_middle_channels=(16 * 2, 32 * 2, 64 * 2, 128 * 2),
        kernel_size=3,
    ):
        super().__init__()
        assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]

        self.encoder = PretrainedLanKernel(device, load=20)

        ###
        self.conv_up_8 = torch.nn.Sequential(
            OrderedDict([("dec-res-8", ResBlock(self.l4_c, self.l3_c, kernel_size))])
        )

        self.conv_up_4 = torch.nn.Sequential(
            OrderedDict([("dec-res-4", ResBlock(self.l3_c, self.l2_c, kernel_size))])
        )

        self.conv_up_2 = torch.nn.Sequential(
            OrderedDict([("dec-res-2", ResBlock(self.l2_c, self.l1_c, kernel_size))])
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

        self.upsample_8_4 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_4_2 = nn.Upsample(scale_factor=2, mode="bilinear")

    def forwardDecoder(self, feature_map_16):
        ##
        # feature_map_16 = self.upsample_16_8(feature_map_16)
        feature_map_up_8 = self.conv_up_8(feature_map_16)
        ##
        # print(feature_map_up_8.shape)
        feature_map_up_8 = self.upsample_8_4(feature_map_up_8)
        feature_map_up_4 = self.conv_up_4(feature_map_up_8)
        ##
        # print(feature_map_up_4.shape)
        feature_map_up_4 = self.upsample_4_2(feature_map_up_4)
        feature_map_up_2 = self.conv_up_2(feature_map_up_4)
        # print(feature_map_up_2.shape)
        # print(feature_map_up_2.shape)
        feature_map_final = self.final(feature_map_up_2)

        return feature_map_final

    def forward(self, lan_emd):
        img, f = self.encoder.get_f(lan_emd)
        # print(f.shape)
        out = self.forwardDecoder(f)
        # print('====',out.shape)
        return out


# x  = torch.rand(1,512).cuda()
# backbone = BackBone().cuda()
# y = backbone(x)
# print(y.shape)
