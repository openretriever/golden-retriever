# pip install ultralytics

import logging

import ray
import requests
import torch
from PIL import Image
from ultralytics import YOLO

from retriever.models.model_base import LangDetectBase

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class YOLOActor(LangDetectBase):
    def __init__(self, model_name="yolo11n.pt", use_gpu=False):
        super().__init__(use_gpu)
        self.device = "cuda" if use_gpu else "cpu"
        self.model = YOLO(model_name).to(self.device)

    def predict(self, image):
        """Assuming `image` is a single PIL.Image objects and `text` is a single string"""
        results = self.model.predict(image, device=self.device)

        """
        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen
            result.save(filename="result.jpg")  # save to disk
        """
        return results


if __name__ == "__main__":
    # Start a Ray cluster
    ray.init()

    print(ray.available_resources())

    # Adjust based on your setup
    use_gpu = torch.cuda.is_available()

    actor_options = {"num_gpus": 1} if use_gpu else {}

    actor = YOLOActor.options(**actor_options).remote(use_gpu=use_gpu)
    logging.info("Actor ready for inference.")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # noqa
    image = Image.open(requests.get(url, stream=True).raw)

    result = ray.get(actor.predict.remote(image))
    logging.info("Inference complete.")

    print(result)

    ray.shutdown()
