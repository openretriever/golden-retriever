import logging
import urllib

import numpy as np
import ray
import requests
import torch
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class DinoV2Actor:
    def __init__(
        self, backbone_name, head_scale_count, head_dataset, head_type, use_gpu=False
    ):
        self.device = "cuda" if use_gpu else "cpu"
        self.backbone_name = backbone_name
        self.head_scale_count = head_scale_count
        self.head_dataset = head_dataset
        self.head_type = head_type
        self.init_model()

    def load_config_from_url(self, url):
        with urllib.request.urlopen(url) as f:
            return f.read().decode()

    def init_model(self):
        # Assuming backbone and segmentation head configurations are loaded similarly
        dinov2_base_url = "https://dl.fbaipublicfiles.com/dinov2"
        head_config_url = f"{dinov2_base_url}/{self.backbone_name}/{self.backbone_name}_{self.head_dataset}_{self.head_type}_config.py"
        head_checkpoint_url = f"{dinov2_base_url}/{self.backbone_name}/{self.backbone_name}_{self.head_dataset}_{self.head_type}_head.pth"

        cfg_str = self.load_config_from_url(head_config_url)
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
        if self.head_type == "ms":
            cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1][
                "img_ratios"
            ][: self.head_scale_count]

        # FIXME - no init_segmentor
        self.model = init_segmentor(cfg, head_checkpoint_url)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, url):
        # Load and process image
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        array = np.array(image)[:, :, ::-1]  # Convert RGB to BGR
        # FIXME - no init_segmentor
        result = inference_segmentor(self.model, array)
        return result


if __name__ == "__main__":
    ray.init()

    # Example configuration (adjust as needed)
    backbone_name = "dinov2_vits14"
    head_scale_count = 3
    head_dataset = "voc2012"
    head_type = "ms"
    use_gpu = torch.cuda.is_available()

    # Create an actor instance
    dinov2_actor = DinoV2Actor.options(num_gpus=1 if use_gpu else 0).remote(
        backbone_name, head_scale_count, head_dataset, head_type, use_gpu
    )

    # URL of the image to process
    url = "https://static01.nyt.com/images/2020/09/08/well/physed-cycle-walk/physed-cycle-walk-videoSixteenByNineJumbo1600-v2.jpg"

    # Perform prediction
    result_future = dinov2_actor.predict.remote(url)
    result = ray.get(result_future)

    print(result)

    # Shutdown Ray
    ray.shutdown()
