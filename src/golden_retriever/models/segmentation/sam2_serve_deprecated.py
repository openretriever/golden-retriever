import logging
import time

import numpy as np
import ray
import torch
from fastapi import FastAPI
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle
from sam2.sam2_image_predictor import SAM2ImagePredictor  # Official SAM2 import

from retriever.models.model_base import BaseModelServer

app = FastAPI()


@serve.deployment(
    name="sam2_model_service",
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.2},
)
class SAM2Service(BaseModelServer):
    def __init__(
        self,
        use_gpu: bool = False,
        model_id: str = "facebook/sam2-hiera-large",
    ):
        """Initialize SAM2 model and processor."""
        super().__init__(use_gpu)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Initialize SAM2 predictor with device
        self.predictor = SAM2ImagePredictor.from_pretrained(
            model_id, device=self.device  # Pass device during initialization
        )

        logging.info(f"SAM2 Model loaded on device: {self.device}")

    async def _process_batch(self, image_batch, points_batch=None, boxes_batch=None):
        """Process a batch of images in sequence."""
        results = []

        # Create a single CUDA context for the batch
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            for idx, image in enumerate(image_batch):
                # Set current image
                self.predictor.set_image(image)

                # Get corresponding points/boxes if provided
                points = points_batch[idx] if points_batch else None
                boxes = boxes_batch[idx] if boxes_batch else None

                # Generate masks
                if points:
                    # Unpack points into coordinates and labels
                    point_coords = np.array([[p[0], p[1]] for p in points])
                    point_labels = np.array(
                        [1] * len(points)
                    )  # 1 for foreground points
                    masks, scores, _ = self.predictor.predict(
                        point_coords=point_coords, point_labels=point_labels
                    )
                elif boxes:
                    masks, scores, _ = self.predictor.predict(boxes=boxes)
                else:
                    masks, scores, _ = self.predictor.predict()

                # Convert outputs to CPU numpy
                # masks = [mask.cpu().numpy() if hasattr(mask, 'cpu') else mask.numpy() for mask in masks]
                # scores = scores.cpu().numpy().tolist() if hasattr(scores, 'cpu') else scores.numpy().tolist()

                results.append({"masks": masks, "scores": scores})

        return results

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def __call__(self, requests):
        """Handle batch requests more efficiently.

        Args:
            requests: List of request dictionaries from Ray Serve's batching
        """
        # Prepare batch inputs
        images = []
        points = []
        boxes = []
        has_points = False
        has_boxes = False

        # FIXME: not sure why requests is a list of lists

        # Handle requests - it's a flat list of dictionaries
        for request_dict in requests:
            # Each request_dict should be a dictionary
            image = request_dict["image"]
            if isinstance(image, str):
                image = Image.open(image)
            images.append(image)

            if "points" in request_dict:
                points.append(request_dict["points"])
                has_points = True
            if "boxes" in request_dict:
                boxes.append(request_dict["boxes"])
                has_boxes = True

        # Process the batch
        results = await self._process_batch(
            image_batch=images,
            points_batch=points if has_points else None,
            boxes_batch=boxes if has_boxes else None,
        )

        return results


if __name__ == "__main__":
    # Initialize Ray and Serve
    ray.init()
    serve.start(http_options={"port": 8200})

    # Create deployment
    sam_deployment = SAM2Service.bind(use_gpu=torch.cuda.is_available())
    sam_handle: DeploymentHandle = serve.run(sam_deployment)

    # Test prediction
    image_path = "tests/images/test_spot_table_cable_1.jpg"
    image = Image.open(image_path)

    # Test with point prompt
    point_request = {
        "image": image,
        "points": [[500, 375]],
    }

    # Test with box prompt
    box_request = {
        "image": image,
        "boxes": [[100, 100, 400, 400]],
    }

    # Make predictions - send requests directly
    point_response = sam_handle.remote(point_request)  # Don't wrap in list
    box_response = sam_handle.remote(box_request)  # Don't wrap in list

    point_result = point_response.result()
    box_result = box_response.result()

    print("Point prompt results:", point_result)
    print("Box prompt results:", box_result)

    # Test batching
    batch_sizes = [1, 2, 4, 8]
    for size in batch_sizes:
        batch_requests = [point_request] * size  # Create list of requests
        start_time = time.time()
        batch_response = sam_handle.remote(
            batch_requests
        )  # Ray Serve will handle batching
        batch_results = batch_response.result()
        end_time = time.time()
        print(f"Batch size {size}: {end_time - start_time:.2f}s")

    serve.shutdown()
