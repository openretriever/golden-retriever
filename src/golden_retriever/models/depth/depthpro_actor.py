# pip install -e external/depth-pro/.
# pip install --upgrade timm # timm-1.0.10

import logging
import os.path
from typing import List, Tuple

import depth_pro
import numpy as np
import ray
import requests
import torch
from depth_pro.depth_pro import DepthProConfig
from depth_pro.utils import extract_exif, fpx_from_f35
from huggingface_hub import snapshot_download
from PIL import Image

from retriever.models.model_base import LangDetectBase

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@ray.remote
class DepthProActor(LangDetectBase):
    def __init__(self, model_name="src/models/checkpoints/depth_pro.pt", use_gpu=False):
        super().__init__(use_gpu)
        self.device = "cuda" if use_gpu else torch.device("cpu")

        if not os.path.exists(model_name):
            repo_id = "apple/DepthPro"
            local_dir = "./src/models/checkpoints"
            snapshot_download(repo_id, local_dir=local_dir)
            model_name = "src/models/checkpoints/depth_pro.pt"

        self.config = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=model_name,
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )

        self.model, self.transform = depth_pro.create_model_and_transforms(
            self.config, device=self.device
        )
        self.model.eval()

    def predict(self, images):
        # images = requests.get(images, stream=True).raw
        def load_rgb_pil(
            img_pil: Image, auto_rotate: bool = True, remove_alpha: bool = True
        ) -> Tuple[np.ndarray, List[bytes], float]:
            """Load an RGB image.

            Args:
            ----
                img_pil: Image
                auto_rotate: Rotate the image based on the EXIF data, default is True.
                remove_alpha: Remove the alpha channel, default is True.

            Returns:
            -------
                img: The image loaded as a numpy array.
                icc_profile: The color profile of the image.
                f_px: The optional focal length in pixels, extracting from the exif data.

            """

            img_exif = extract_exif(img_pil)
            icc_profile = img_pil.info.get("icc_profile", None)

            # Rotate the image.
            if auto_rotate:
                exif_orientation = img_exif.get("Orientation", 1)
                if exif_orientation == 3:
                    img_pil = img_pil.transpose(Image.ROTATE_180)
                elif exif_orientation == 6:
                    img_pil = img_pil.transpose(Image.ROTATE_270)
                elif exif_orientation == 8:
                    img_pil = img_pil.transpose(Image.ROTATE_90)
                elif exif_orientation != 1:
                    LOGGER.warning(f"Ignoring image orientation {exif_orientation}.")

            img = np.array(img_pil)
            # Convert to RGB if single channel.
            if img.ndim < 3 or img.shape[2] == 1:
                img = np.dstack((img, img, img))

            if remove_alpha:
                img = img[:, :, :3]

            LOGGER.debug(f"\tHxW: {img.shape[0]}x{img.shape[1]}")

            # Extract the focal length from exif data.
            f_35mm = img_exif.get(
                "FocalLengthIn35mmFilm",
                img_exif.get(
                    "FocalLenIn35mmFilm", img_exif.get("FocalLengthIn35mmFormat", None)
                ),
            )
            if f_35mm is not None and f_35mm > 0:
                LOGGER.debug(f"\tfocal length @ 35mm film: {f_35mm}mm")
                f_px = fpx_from_f35(img.shape[1], img.shape[0], f_35mm)
            else:
                f_px = None

            return img, icc_profile, f_px

        image, _, f_px = load_rgb_pil(images)
        image = self.transform(image)

        # Run inference.
        prediction = self.model.infer(image, f_px=f_px)

        # depth = prediction["depth"]  # Depth in [m].
        # focallength_px = prediction["focallength_px"]  # Focal length in pixels.

        return prediction


if __name__ == "__main__":
    ray.init()
    # ray.init("ray://localhost:10002")
    print(ray.available_resources())

    # Adjust based on your setup
    use_gpu = torch.cuda.is_available()

    actor_options = {"num_gpus": 1} if use_gpu else {}

    # Create an actor
    actor = DepthProActor.options(**actor_options).remote(use_gpu=use_gpu)

    # Example usage
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # noqa
    image = Image.open(requests.get(url, stream=True).raw)

    results = ray.get(actor.predict.remote(image))

    print(results)

    ray.shutdown()
