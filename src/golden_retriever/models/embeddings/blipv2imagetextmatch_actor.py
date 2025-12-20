# Configure logging for Ray and other components
import logging

import ray
import requests
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

logging.basicConfig(level=logging.INFO)


@ray.remote
class Blipv2ImageTextMatchingActor:
    def __init__(
        self,
        model_name: str = "blip2_image_text_matching",
        model_type: str = "pretrain",
        use_gpu: bool = False,
    ):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        (
            self.model,
            self.vis_processors,
            self.text_processors,
        ) = load_model_and_preprocess(
            name=model_name,
            model_type=model_type,
            is_eval=True,
            device=self.device,
        )

    def predict(self, image: Image, txt: str) -> float:
        """
        Compute the cosine similarity between the image and the prompt.

        Args:
            image (PIL.Image): The input image as a PIL image.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
            float: The softmax scores for the image-text matching task (image and text are matched with a probability of)
        """
        img = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with torch.inference_mode():
            itc_score = self.model(
                {"image": img, "text_input": txt}, match_head="itc"
            ).item()
            itm_output = self.model({"image": img, "text_input": txt}, match_head="itm")
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)

        return itc_score, itm_scores


if __name__ == "__main__":
    # Initialize Ray, replace with your cluster's address or local setup
    ray.init(num_gpus=1)  # Add arguments as necessary, e.g., address, num_gpus

    # Variable to control GPU usage
    use_gpu = torch.cuda.is_available()
    actor_handle = Blipv2ImageTextMatchingActor.remote(use_gpu=use_gpu)
    print("Actor initialized")

    # Example image URL and input point
    image_url = (
        "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    )
    image_pil = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    # Example text
    txt = "a car on the road"

    # Perform prediction
    itc_score, itm_scores = ray.get(actor_handle.predict.remote(image_pil, txt))
    print(f"Image-Text Cosine Similarity: {itc_score}")
    print(f"Image-Text Matching Scores: {itm_scores}")

    ray.shutdown()
