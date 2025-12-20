import logging

import ray
import requests
import torch
from PIL import Image
from transformers import pipeline

from retriever.models.model_base import LangDetectBase

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class DepthEstimationActor(LangDetectBase):
    def __init__(self, model_name="Intel/zoedepth-nyu-kitti", use_gpu=False):
        super().__init__(use_gpu)
        self.device = "cuda" if use_gpu else "cpu"
        self.model = pipeline(
            task="depth-estimation", model=model_name, device=self.device
        )

    def predict(self, images):
        prediction = self.model(images)
        return prediction


if __name__ == "__main__":
    ray.init()
    # ray.init("ray://localhost:10002")
    print(ray.available_resources())

    # Adjust based on your setup
    use_gpu = torch.cuda.is_available()

    actor_options = {"num_gpus": 1} if use_gpu else {}

    # Create an actor
    actor = DepthEstimationActor.options(**actor_options).remote(use_gpu=use_gpu)

    # Example usage
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # noqa
    image = Image.open(requests.get(url, stream=True).raw)

    results = ray.get(actor.predict.remote(image))
    print(results)
    ray.shutdown()
