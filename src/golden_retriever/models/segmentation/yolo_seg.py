import logging

import ray
import requests
import torch
from PIL import Image

from retriever.models.detection.yolo_actor import YOLOActor

"""
Example of using YOLO for Segmentation

"""

if __name__ == "__main__":
    # Start a Ray cluster
    ray.init()

    print(ray.available_resources())

    # Adjust based on your setup
    use_gpu = torch.cuda.is_available()

    actor_options = {"num_gpus": 1} if use_gpu else {}

    actor = YOLOActor.options(**actor_options).remote(
        model_name="yolo11n-seg.pt", use_gpu=use_gpu
    )
    logging.info("Actor ready for inference.")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # noqa
    image = Image.open(requests.get(url, stream=True).raw)

    result = ray.get(actor.predict.remote(image))
    logging.info("Inference complete.")

    print(result)

    ray.shutdown()
