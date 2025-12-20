import torch
import torch.nn.functional as F
from lepp.clip_revised.clip import tokenize
from lepp.kernel_backbone_mlp_conv import BackBone


def encode_text(x, clip_rn50, device):
    with torch.no_grad():
        tokens = tokenize([x]).to(device)
        text_feat, text_emb = clip_rn50.encode_text_with_embeddings(tokens)
    text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
    return text_feat, text_emb, text_mask


class PretrainedLanKernel:
    def __init__(self, device, load=0):
        self.device = device
        self.kernel_backbone = BackBone(
            mlp_dim=(1024, 256, 128),
            in_channel=1,
            out_channel=6,
            n_middle_channels=(16 * 2, 32 * 2, 64 * 2, 128 * 2),
            kernel_size=3,
            lan_emd=True,
        ).to(device)
        print(
            "backbone",
            sum(
                p.numel() for p in self.kernel_backbone.parameters() if p.requires_grad
            ),
        )
        self.optim = torch.optim.Adam(
            self.kernel_backbone.parameters(), lr=1e-4, weight_decay=1e-8
        )
        if load != False:
            path = "./crop_lan_checkpoints/{}.pt".format(load)
            print("load {}".format(path))
            self.load(path)

    def train(self, lan, crop, clip_rn50, backprop=True):
        self.kernel_backbone.train()
        crop = torch.from_numpy(crop).to(torch.float).to(self.device)
        crop = crop.unsqueeze(dim=0).permute(0, 3, 1, 2)
        lan_emd, _, _ = encode_text(lan[0], clip_rn50, self.device)
        lan_emd = lan_emd.to(torch.float).to(self.device)
        img_from_lan, _ = self.kernel_backbone(lan_emd)
        loss = F.mse_loss(input=img_from_lan, target=crop, reduction="mean")
        if backprop:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return float(loss.item())

    def get_f(self, lan_emd):
        self.kernel_backbone.eval()
        with torch.no_grad():
            img_from_lan, f = self.kernel_backbone(lan_emd)
        return img_from_lan, f

    def eval(self, lan, crop, clip_rn50):
        self.kernel_backbone.eval()
        crop = torch.from_numpy(crop).to(torch.float).to(self.device)
        crop = crop.unsqueeze(dim=0).permute(0, 3, 1, 2)
        lan_emd, _, _ = encode_text(lan[0], clip_rn50, self.device)
        lan_emd = lan_emd.to(torch.float).to(self.device)
        with torch.no_grad():
            img_from_lan, _ = self.kernel_backbone(lan_emd)
            loss = F.mse_loss(input=img_from_lan, target=crop, reduction="mean")
        return float(loss.item())

    def load(self, path1):
        self.kernel_backbone.eval()
        self.kernel_backbone.load_state_dict(
            torch.load(path1, map_location=self.device)
        )

    def save(self, path1):
        self.kernel_backbone.eval()
        torch.save(self.kernel_backbone.state_dict(), path1)


# run(args)
