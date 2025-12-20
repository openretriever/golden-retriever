# pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# Configure logging for Ray and other components
import logging
import os

import numpy as np
import ray
import requests
import torch
from huggingface_hub import snapshot_download
from mobile_sam import SamPredictor, sam_model_registry
from PIL import Image

logging.basicConfig(level=logging.INFO)


@ray.remote
class MobileSAMActor:
    def __init__(
        self,
        model_type: str = "vit_t",
        model_name="src/models/checkpoints/mobile_sam.pt",
        use_gpu: bool = False,
    ):
        if not os.path.exists(model_name):
            repo_id = "dhkim2810/MobileSAM"
            local_dir = "./src/models/checkpoints"
            snapshot_download(repo_id, local_dir=local_dir)
            model_name = "src/models/checkpoints/mobile_sam.pt"

        self.device = "cuda" if use_gpu else "cpu"
        self.model = sam_model_registry[model_type](checkpoint=model_name)
        self.model.to(device=self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)

    def predict(
        self, image: Image, bbox: list[int], multimask_output: bool = False
    ) -> np.ndarray:
        """Segments the object in the given bounding box from the image.

        Args:
            image (PIL.Image): The input image as a PIL image.
            bbox (List[int]): The bounding box as a numpy array in the
                format [x1, y1, x2, y2].
            multimask_output (bool): Whether to return multiple masks or not.

        Returns:
            np.ndarray: The segmented object as a numpy array (boolean mask). The mask
                is the same size as the bbox, cropped out of the image.
            np.ndarray: The quality of the mask.
            np.ndarray: The low resolution mask.

        """
        with torch.inference_mode():
            image = np.array(image)
            self.predictor.set_image(image)
            masks, quality, low_res_masks_np = self.predictor.predict(
                box=np.array(bbox), multimask_output=multimask_output
            )

        return masks, quality, low_res_masks_np


if __name__ == "__main__":
    # Initialize Ray, replace with your cluster's address or local setup
    ray.init(num_gpus=1)  # Add arguments as necessary, e.g., address, num_gpus

    # Variable to control GPU usage
    use_gpu = torch.cuda.is_available()

    # Options dictionary for dynamic resource allocation
    actor_options = {"num_gpus": 1} if use_gpu else {}

    # Create an actor instance
    sam_actor = MobileSAMActor.options(**actor_options).remote(use_gpu=use_gpu)

    # Example image URL and input point
    image_url = (
        "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    )
    image_pil = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    input_points = [450, 600, 450, 600]  # Example input point for segmentation

    # Predict masks and scores
    masks, scores, _ = ray.get(sam_actor.predict.remote(image_pil, input_points))

    print("Scores:", scores)
    # Add your logic here to work with masks and scores

    ray.shutdown()
