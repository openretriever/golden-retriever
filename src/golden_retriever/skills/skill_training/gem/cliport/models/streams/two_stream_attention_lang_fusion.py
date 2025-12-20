import cliport.models as models
import cliport.models.core.fusion as fusion
import numpy as np
import torch
import torch.nn.functional as F
from cliport.models.core.attention import Attention
from lepp.kernel_backbone_mlp_conv import ResBlock


class TwoStreamAttentionLangFusion(Attention):
    """Two Stream Language-Conditioned Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg["train"]["attn_stream_fusion_type"]
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(
            self.in_shape, 1, self.cfg, self.device, self.preprocess
        ).to(self.device)
        self.attn_stream_two = stream_two_model(
            self.in_shape, 1, self.cfg, self.device, self.preprocess
        ).to(self.device)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(
            f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}"
        )

    def attend(self, x, l):
        x1 = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, l)
        x = self.fusion(x1, x2)
        return x

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        in_data = np.pad(inp_img, self.padding, mode="constant")
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(
            dtype=torch.float, device=self.device
        )  # [B W H 6]

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = []
        for x in in_tens:
            lgts = self.attend(x, lang_goal)
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0] : c1[0], c0[1] : c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output


class TwoStreamAttentionLangFusionLat(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg["train"]["attn_stream_fusion_type"]
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def attend(self, x, l):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, l)
        x = self.fusion(x1, x2)
        return x


class TwoStreamAttentionLangFusionLatLEPP(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(
        self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device, out_channel
    ):
        self.fusion_type = cfg["train"]["attn_stream_fusion_type"]
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)
        self.conv = ResBlock(1, out_channel, 3, last_relu=False).to(device)

    def attend(self, x, l):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, l)
        x = self.fusion(x1, x2)
        x = self.conv(x)
        return x

    def forward(self, in_tensor, lang_goal):
        """Forward pass."""

        lgts = self.attend(in_tensor, lang_goal)

        return lgts


class TwoStreamAttentionLangFusionLatLEPPPostLinearMul(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(
        self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device, out_channel
    ):
        self.fusion_type = cfg["train"]["attn_stream_fusion_type"]
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)
        self.conv = ResBlock(1, out_channel, 3, last_relu=False).to(device)
        shape_out = 3
        self.shape_conv1 = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1).to(device)
        self.shape_relu1 = torch.nn.ReLU().to(device)
        self.shape_conv2 = torch.nn.Conv2d(3, shape_out, kernel_size=3, padding=1).to(
            device
        )

        self.conv_clip = torch.nn.Conv2d(
            1 + shape_out, out_channel, kernel_size=9, padding=4
        ).to(device)

    def attend(self, x, l):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, l)
        x = self.fusion(x1, x2)
        x = self.conv(x)
        return x

    def forward(self, in_tensor, clip_feature, lang_goal=None, dist="transporter"):
        """Forward pass."""
        in_tensor = self.preprocess(in_tensor, dist=dist)
        lgts = self.attend(in_tensor, lang_goal)

        device = in_tensor.device
        depth = in_tensor[:1, -1:, ...].detach().clone().to(device)
        shape_feature = self.shape_conv2(self.shape_relu1(self.shape_conv1(depth)))
        clip_feature = torch.cat(
            [clip_feature, shape_feature], axis=1
        )  # encode some shape info
        clip_feature = self.conv_clip(clip_feature)
        lgts *= clip_feature
        return lgts
