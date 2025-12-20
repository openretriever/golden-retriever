import logging

import ray
import requests
import torch
from PIL import Image

from retriever.models.depth.depth_pipeline_actor import DepthEstimationActor

# Configure logging
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    ray.init()
    # ray.init("ray://localhost:10002")
    print(ray.available_resources())

    # Adjust based on your setup
    use_gpu = torch.cuda.is_available()

    actor_options = {"num_gpus": 1} if use_gpu else {}

    # Create an actor
    # actor = DepthEstimationActor.options(**actor_options).remote(model_name="depth-anything/Depth-Anything-V2-Small-hf",use_gpu=use_gpu)
    # actor = DepthEstimationActor.options(**actor_options).remote(model_name="depth-anything/Depth-Anything-V2-Base-hf",use_gpu=use_gpu)
    actor = DepthEstimationActor.options(**actor_options).remote(
        model_name="depth-anything/Depth-Anything-V2-Large-hf", use_gpu=use_gpu
    )

    # Example usage
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # noqa
    image = Image.open(requests.get(url, stream=True).raw)

    results = ray.get(actor.predict.remote(image))
    print(results)
    ray.shutdown()
