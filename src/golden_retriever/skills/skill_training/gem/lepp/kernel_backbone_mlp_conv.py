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


# x = torch.rand(1,1,100,100)
# resb1 = ResBlock(input_channels=1, hidden_dim=10,kernel_size=3)
# y = resb1(x)
# print(y.shape)

# x = torch.rand(1,100)
# fc = torch.nn.Linear(in_features=100,out_features=50,bias=True)
# y = fc(x)
# print(y.shape)


class BackBone(torch.nn.Module):
    def __init__(
        self,
        mlp_dim=(512, 256, 128),
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=True,
    ):
        super().__init__()
        assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]
        self.mlp = nn.Sequential(
            torch.nn.Linear(mlp_dim[0], mlp_dim[1], bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mlp_dim[1], mlp_dim[2], bias=True),
            torch.nn.ReLU(inplace=True),
        )

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
        self.lan_emd = lan_emd
        if lan_emd is False:
            print("use initial")
            self.initial = torch.nn.parameter.Parameter(
                torch.zeros(1, mlp_dim[0]) + 0.5, requires_grad=True
            )

    def _init(self):
        pass

    def forward_mlp(self, x):
        # print(x.shape,'****')
        if self.lan_emd:
            x = self.mlp(x)
        else:
            x = self.mlp(self.initial)

        return x

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
        # print(feature_map_up_2.shape)
        feature_map_final = self.final(feature_map_up_2)

        return feature_map_final

    def forward(self, x):
        x = self.forward_mlp(x)

        x = x.repeat(x.shape[-1], 1)
        # similar performance with the expansion function below
        # x = torch.einsum("ij,ik->ik",x.permute(1,0),x)
        # print(x.shape)
        x = x.unsqueeze(dim=0).unsqueeze(dim=0)
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
        # print('====',out.shape)
        return out


class BackBoneM(torch.nn.Module):
    def __init__(
        self,
        mlp_dim=(512, 256, 128, 128),
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128, 256, 512),
        kernel_size=3,
        lan_emd=True,
    ):
        super().__init__()
        assert len(n_middle_channels) == 6
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]
        self.l5_c = n_middle_channels[4]
        self.l6_c = n_middle_channels[5]
        self.mlp = nn.Sequential(
            torch.nn.Linear(mlp_dim[0], mlp_dim[1], bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mlp_dim[1], mlp_dim[2], bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mlp_dim[2], mlp_dim[3], bias=True),
        )

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
                    ("enc-res-16", ResBlock(self.l4_c, self.l5_c, kernel_size)),
                ]
            )
        )
        self.conv_down_32 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-32", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-32", ResBlock(self.l5_c, self.l6_c, kernel_size)),
                ]
            )
        )
        self.conv_down_64 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("env-pool-64", nn.MaxPool2d(kernel_size=2)),
                    ("enc-res-64", ResBlock(self.l6_c, self.l6_c, kernel_size)),
                ]
            )
        )

        self.conv_up_32 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-32", ResBlock(2 * self.l6_c, self.l5_c, kernel_size))]
            )
        )

        self.conv_up_16 = torch.nn.Sequential(
            OrderedDict(
                [("dec-res-16", ResBlock(2 * self.l5_c, self.l4_c, kernel_size))]
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

        self.upsample_64_32 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_32_16 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_16_8 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_8_4 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample_4_2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.lan_emd = lan_emd
        if lan_emd is False:
            print("use initial")
            self.initial = torch.nn.parameter.Parameter(
                torch.zeros(1, mlp_dim[0]) + 0.5, requires_grad=True
            )

    def _init(self):
        pass

    def forward_mlp(self, x):
        # print(x.shape,'****')
        if self.lan_emd:
            x = self.mlp(x)
        else:
            x = self.mlp(self.initial)

        return x

    def forwardEncoder(self, x):
        feature_map_1 = self.conv_down_1(x)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        feature_map_32 = self.conv_down_32(feature_map_16)
        feature_map_64 = self.conv_down_64(feature_map_32)
        return (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            feature_map_32,
            feature_map_64,
        )

    def forwardDecoder(
        self,
        feature_map_1,
        feature_map_2,
        feature_map_4,
        feature_map_8,
        feature_map_16,
        feature_map_32,
        feature_map_64,
    ):
        concat_32 = torch.cat(
            (feature_map_32, self.upsample_64_32(feature_map_64)), dim=1
        )
        feature_map_up_32 = self.conv_up_32(concat_32)

        concat_16 = torch.cat(
            (feature_map_16, self.upsample_32_16(feature_map_up_32)), dim=1
        )
        feature_map_up_16 = self.conv_up_16(concat_16)

        concat_8 = torch.cat(
            (feature_map_8, self.upsample_16_8(feature_map_up_16)), dim=1
        )
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat(
            (feature_map_4, self.upsample_8_4(feature_map_up_8)), dim=1
        )
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat(
            (feature_map_2, self.upsample_4_2(feature_map_up_4)), dim=1
        )
        feature_map_up_2 = self.conv_up_2(concat_2)
        # print(feature_map_up_2.shape)
        feature_map_final = self.final(feature_map_up_2)

        return feature_map_final

    def forward(self, x):
        x = self.forward_mlp(x)

        x = x.repeat(x.shape[-1], 1)
        # similar performance with the expansion function below
        # x = torch.einsum("ij,ik->ik",x.permute(1,0),x)
        # print(x.shape)
        x = x.unsqueeze(dim=0).unsqueeze(dim=0)
        (
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            feature_map_32,
            feature_map_64,
        ) = self.forwardEncoder(x)
        out = self.forwardDecoder(
            feature_map_1,
            feature_map_2,
            feature_map_4,
            feature_map_8,
            feature_map_16,
            feature_map_32,
            feature_map_64,
        )
        # print('====',out.shape)
        return out


class BackBoneDiffusion(torch.nn.Module):
    def __init__(
        self,
        mlp_dim=(512, 256, 128),
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=True,
        n_diff=3,
    ):
        super().__init__()
        assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]
        self.mlp = nn.Sequential(
            torch.nn.Linear(mlp_dim[0], mlp_dim[1], bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mlp_dim[1], mlp_dim[2], bias=True),
            torch.nn.ReLU(inplace=True),
        )

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
                [("dec-res-1", ResBlock(2 * self.l1_c, in_channel, kernel_size))]
            )
        )

        self.final = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "dec-final",
                        torch.nn.Conv2d(
                            in_channels=in_channel,
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
        self.lan_emd = lan_emd
        self.n_diff = n_diff

        if lan_emd is False:
            self.initial = torch.nn.parameter.Parameter(
                torch.zeros(1, mlp_dim[0]) + 0.5, requires_grad=True
            )

    def _init(self):
        pass

    def forward_mlp(self, x):
        # print(x.shape,'****')
        if self.lan_emd:
            x = self.mlp(x)
        else:
            x = self.mlp(self.initial)
            print("use initial")
        return x

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
        # print(feature_map_up_2.shape)

        concat_1 = torch.cat(
            (feature_map_1, self.upsample_2_1(feature_map_up_2)), dim=1
        )
        feature_map_up_1 = self.conv_up_1(concat_1)

        return feature_map_up_1

    def forward(self, x):
        x = self.forward_mlp(x)

        x = x.repeat(x.shape[-1], 1)
        # similar performance with the expansion function below
        # x = torch.einsum("ij,ik->ik",x.permute(1,0),x)
        # print(x.shape)
        x = x.unsqueeze(dim=0).unsqueeze(dim=0)

        for _ in range(self.n_diff):
            (
                feature_map_1,
                feature_map_2,
                feature_map_4,
                feature_map_8,
                feature_map_16,
            ) = self.forwardEncoder(x)
            x = self.forwardDecoder(
                feature_map_1,
                feature_map_2,
                feature_map_4,
                feature_map_8,
                feature_map_16,
            )
        out = self.final(x)
        # print('====',out.shape)
        return out


class CropUnet(torch.nn.Module):
    def __init__(
        self,
        in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
    ):
        super().__init__()
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
                [("dec-res-1", ResBlock(2 * self.l1_c, self.l1_c, kernel_size))]
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
        # print(feature_map_up_2.shape)
        feature_map_final = self.final(feature_map_up_1)

        return feature_map_final

    def forward(self, x):
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
        # print('====',out.shape)
        return out


class CropUnetLanUNetADD(torch.nn.Module):
    def __init__(
        self,
        mlp_dim=(512, 256, 128),
        obs_in_channel=6,
        lan_in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=True,
    ):
        super().__init__()

        self.crop_net = CropUnet(
            in_channel=obs_in_channel,
            out_channel=out_channel,
            n_middle_channels=n_middle_channels,
            kernel_size=kernel_size,
        )
        self.lan_net = BackBone(
            mlp_dim=mlp_dim,
            in_channel=lan_in_channel,
            out_channel=out_channel,
            n_middle_channels=n_middle_channels,
            kernel_size=kernel_size,
            lan_emd=lan_emd,
        )
        self.lan_emd = lan_emd

        if lan_emd is False:
            self.initial = torch.nn.parameter.Parameter(
                torch.zeros(1, mlp_dim[0]) + 0.5, requires_grad=True
            )

    def _init(self):
        pass

    def forward_mlp(self, x):
        # print(x.shape,'****')
        if self.lan_emd:
            x = self.mlp(x)
        else:
            x = self.mlp(self.initial)
            print("use initial")
        return x

    def forward(self, x, lan):
        lan_feature = self.lan_net(lan)

        crop_feature = self.crop_net(x)
        out = lan_feature + crop_feature
        # print('====',out.shape)
        return out


class CropUnetLanUNetCAT(torch.nn.Module):
    def __init__(
        self,
        mlp_dim=(512, 256, 128),
        obs_in_channel=6,
        lan_in_channel=1,
        lan_out_channel=3,
        crop_out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=True,
    ):
        super().__init__()

        self.crop_net = CropUnet(
            in_channel=obs_in_channel,
            out_channel=lan_out_channel,
            n_middle_channels=n_middle_channels,
            kernel_size=kernel_size,
        )
        self.lan_net = BackBone(
            mlp_dim=mlp_dim,
            in_channel=lan_in_channel,
            out_channel=crop_out_channel,
            n_middle_channels=n_middle_channels,
            kernel_size=kernel_size,
            lan_emd=lan_emd,
        )
        self.lan_emd = lan_emd

        if lan_emd is False:
            self.initial = torch.nn.parameter.Parameter(
                torch.zeros(1, mlp_dim[0]) + 0.5, requires_grad=True
            )

    def _init(self):
        pass

    def forward_mlp(self, x):
        # print(x.shape,'****')
        if self.lan_emd:
            x = self.mlp(x)
        else:
            x = self.mlp(self.initial)
            print("use initial")
        return x

    def forward(self, x, lan):
        lan_feature = self.lan_net(lan)

        crop_feature = self.crop_net(x)
        out = torch.cat([lan_feature, crop_feature], dim=1)
        # print('====',out.shape)
        return out


class CropLanUnetOld(torch.nn.Module):
    def __init__(
        self,
        mlp_dim=(512, 256, 128, 64),
        obs_in_channel=6,
        lan_in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=True,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            torch.nn.Linear(mlp_dim[0], mlp_dim[1], bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mlp_dim[1], mlp_dim[2], bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mlp_dim[2], mlp_dim[3], bias=True),
            torch.nn.ReLU(inplace=True),
        )

        self.crop_net = CropUnet(
            in_channel=obs_in_channel + lan_in_channel,
            out_channel=out_channel,
            n_middle_channels=n_middle_channels,
            kernel_size=kernel_size,
        )
        self.lan_emd = lan_emd

        if lan_emd is False:
            self.initial = torch.nn.parameter.Parameter(
                torch.zeros(1, mlp_dim[0]) + 0.5, requires_grad=True
            )

    def _init(self):
        pass

    def forward_mlp(self, x):
        # print(x.shape,'****')
        if self.lan_emd:
            x = self.mlp(x)
        else:
            x = self.mlp(self.initial)
            print("use initial")
        return x

    def forward(self, x, lan):
        lan_feature = self.forward_mlp(lan)
        lan_feature = lan_feature.repeat(lan_feature.shape[-1], 1)
        lan_feature = lan_feature.unsqueeze(dim=0).unsqueeze(dim=0)
        cat_feature = torch.cat([lan_feature, x], dim=1)
        out = self.crop_net(cat_feature)
        # print('====',out.shape)
        return out


class CropLanUnet(torch.nn.Module):
    def __init__(
        self,
        mlp_dim=(512, 256, 128, 64),
        obs_in_channel=6,
        lan_in_channel=1,
        out_channel=3,
        n_middle_channels=(16, 32, 64, 128),
        kernel_size=3,
        lan_emd=True,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            torch.nn.Linear(mlp_dim[0], mlp_dim[1], bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mlp_dim[1], mlp_dim[2], bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(mlp_dim[2], mlp_dim[3], bias=True),
            torch.nn.ReLU(inplace=True),
        )

        self.crop_net = CropUnet(
            in_channel=obs_in_channel + lan_in_channel,
            out_channel=out_channel,
            n_middle_channels=n_middle_channels,
            kernel_size=kernel_size,
        )
        self.lan_emd = lan_emd

        if lan_emd is False:
            self.initial = torch.nn.parameter.Parameter(
                torch.zeros(1, mlp_dim[0]) + 0.5, requires_grad=True
            )

    def _init(self):
        pass

    def forward_mlp(self, x):
        # print(x.shape,'****')
        if self.lan_emd:
            x = self.mlp(x)
        else:
            x = self.mlp(self.initial)
            print("use initial")
        return x

    def forward(self, x, lan):
        lan_feature = self.forward_mlp(lan)
        lan_feature = lan_feature[..., None, None].repeat(1, 1, x.shape[2], x.shape[3])
        cat_feature = torch.cat([lan_feature, x], dim=1)
        out = self.crop_net(cat_feature)
        # print('====',out.shape)
        return out


# x  = torch.rand(1,512).cuda()
# backbone = BackBone().cuda()
# y = backbone(x)
# print(y.shape)
