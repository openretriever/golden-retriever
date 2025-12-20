import logging

import ray
import requests
import torch
from PIL import Image

from retriever.models.detection.yolo_actor import YOLOActor

"""
Example of using YOLO for OBB detection

Oriented Bounding Boxes (OBB) include an additional angle to enhance object localization accuracy in images. Unlike
regular bounding boxes, which are axis-aligned rectangles, OBBs can rotate to fit the orientation of the object better.
This is particularly useful for applications requiring precise object placement, such as aerial or satellite imagery
"""


if __name__ == "__main__":
    # Start a Ray cluster
    ray.init()

    print(ray.available_resources())

    # Adjust based on your setup
    use_gpu = torch.cuda.is_available()

    actor_options = {"num_gpus": 1} if use_gpu else {}

    actor = YOLOActor.options(**actor_options).remote(
        model_name="yolo11n-obb.pt", use_gpu=use_gpu
    )
    logging.info("Actor ready for inference.")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # noqa
    image = Image.open(requests.get(url, stream=True).raw)

    result = ray.get(actor.predict.remote(image))
    logging.info("Inference complete.")

    print(result)

    ray.shutdown()
