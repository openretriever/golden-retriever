"""
Usage:
1. Start the server:
   python -m src.models.segmentation.molmo_sam2_server

2. In another terminal, run the client:
   python -m src.models.segmentation.molmo_sam2_client predict "./tests/images/test_img_cable_with_knots.jpg" "Point at the cable knot"

Output files will be saved in ./src/models/tmp/:
- molmo_sam2_serve_points.png: Visualization of the detected points
- molmo_sam2_serve_segmentation.png: Final segmentation result
"""


import asyncio
import base64
import io
import os

import ray
import typer
from fastapi import FastAPI
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle

from retriever.models.model_base import BaseModelServer
from retriever.models.segmentation.sam2_actor import SAM2Actor
from retriever.models.vlms.molmo_quantized_actor import MolmoQuantizedActor, draw_points

app = FastAPI()


@serve.deployment(
    name="molmo_sam2_service",
    num_replicas=1,
    max_ongoing_requests=8,
    # NOTE: make sure the Service has minimum access to GPUs to hold CUDA tensors
    ray_actor_options={"num_gpus": 0.01},
)
class MolmoSAM2Service(BaseModelServer):
    def __init__(self, use_gpu: bool = False):
        super().__init__(use_gpu)
        print("Using GPU:", use_gpu)

        # Initialize the actors with GPU options if available
        molmo_actor_options = {"num_gpus": 0.9} if use_gpu else {}
        sam_actor_options = {"num_gpus": 0.9} if use_gpu else {}

        self.molmo_actor = MolmoQuantizedActor.options(**molmo_actor_options).remote(
            use_gpu=use_gpu
        )
        self.sam_actor = SAM2Actor.options(**sam_actor_options).remote(use_gpu=use_gpu)

    async def _process_image(self, image_data):
        """Convert base64 image data to PIL Image"""
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))
        return image_data

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def __call__(self, requests):
        """Handle batched requests"""
        responses = []
        for request in requests:
            # Handle both Starlette Request objects and dictionaries
            if hasattr(request, "json"):
                # If it's a Starlette Request object
                request_data = await request.json()
            else:
                # If it's already a dictionary
                request_data = request

            try:
                image_data = request_data.get("image")
                prompt = request_data.get("prompt")

                if not image_data or not prompt:
                    raise ValueError("Missing 'image' or 'prompt' in request")

                result = await self.predict(image_data, prompt)
                responses.append(result)
            except Exception as e:
                print(f"Error processing request: {e}")
                responses.append({"error": str(e)})

        return responses

    async def predict(self, image_data: str, prompt: str):
        """Main prediction method"""
        try:
            print("Starting prediction...")
            image = await self._process_image(image_data)

            tmp_dir = os.path.join("src", "models", "tmp")
            os.makedirs(tmp_dir, exist_ok=True)

            # Get points from Molmo
            print("Getting points from Molmo...")
            molmo_result = ray.get(
                self.molmo_actor.predict.remote(image, prompt, points=True)
            )

            # Save intermediate result with points
            drawn_image = draw_points(image, molmo_result["points"])
            points_path = os.path.join(tmp_dir, "molmo_sam2_serve_points.png")
            drawn_image.save(points_path)
            print(f"Points visualization saved to: {points_path}")

            # Get segmentation from SAM
            print("Getting segmentation from SAM...")
            sam_result = ray.get(
                self.sam_actor.predict.remote(
                    image, input=molmo_result["points"], type="point"
                )
            )

            # Save the segmentation result
            result_path = os.path.join(tmp_dir, "molmo_sam2_serve_segmentation.png")
            # Plot and save the result
            sam_result.plot(save=True, filename=result_path)
            print(f"Segmentation result saved to: {result_path}")

            # Read the saved image and convert to base64
            with open(result_path, "rb") as f:
                result_b64 = base64.b64encode(f.read()).decode()

            print("Prediction complete.")
            return {"points": molmo_result["points"], "segmentation_image": result_b64}
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            import traceback

            traceback.print_exc()
            raise


def start_server(
    port: int = typer.Option(8100, help="Port number for the service"),
    use_gpu: bool = typer.Option(True, help="Whether to use GPU for inference"),
    max_ongoing_requests: int = typer.Option(
        8, help="Maximum number of concurrent queries"
    ),
):
    """Start the MolmoSAM2 Service with specified configuration"""
    # Shutdown any existing Ray instance
    if ray.is_initialized():
        ray.shutdown()

    ray.init()
    print(ray.available_resources())  # Add this after ray.init()
    serve.start(http_options={"port": port})

    # Deploy the service
    deployment = MolmoSAM2Service.options(
        max_ongoing_requests=max_ongoing_requests
    ).bind(use_gpu=use_gpu)

    handle: DeploymentHandle = serve.run(deployment)
    print(f"MolmoSAM2Service is running on port {port}")
    print(f"Deployment name: {handle.deployment_name}")

    # Keep the service running
    try:
        while True:
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))
    except KeyboardInterrupt:
        print("\nShutting down service...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    typer.run(start_server)
