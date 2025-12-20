import base64
import io

import ray
import starlette.requests as requests
from fastapi import FastAPI
from PIL import Image
from ray import serve

from retriever.models.model_base import BaseModelServer
from retriever.models.segmentation.sam2_actor import SAM2Actor
from retriever.models.vlms.molmo_quantized_actor import MolmoQuantizedActor

app = FastAPI()


@serve.deployment(
    name="molmo_service",
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},  # Adjust based on your needs
)
class MolmoService(BaseModelServer):
    def __init__(self, use_gpu: bool = False):
        super().__init__(use_gpu)
        self.device = "cuda" if use_gpu else "cpu"
        self.molmo_actor = MolmoQuantizedActor.remote(use_gpu=use_gpu)
        self.sam_actor = SAM2Actor.remote(use_gpu=use_gpu)

    async def _process_image(self, image_data):
        """Convert base64 image data to PIL Image"""
        if isinstance(image_data, str):
            # Assuming base64 encoded image
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))
        return image_data

    async def predict(self, image_data: str, prompt: str):
        """Main prediction method"""
        image = await self._process_image(image_data)

        # Get points from Molmo
        molmo_result = ray.get(
            self.molmo_actor.predict.remote(image, prompt, points=True)
        )

        # Get segmentation from SAM
        sam_result = ray.get(
            self.sam_actor.predict.remote(
                image, input=molmo_result["points"], type="point"
            )
        )

        # Convert result image to base64
        buffered = io.BytesIO()
        sam_result.save(buffered, format="PNG")
        result_b64 = base64.b64encode(buffered.getvalue()).decode()

        return {"points": molmo_result["points"], "segmentation_image": result_b64}

    async def __call__(self, request: requests.Request):
        """Handle incoming HTTP requests"""
        data = await request.json()
        image_data = data["image"]
        prompt = data["prompt"]

        result = await self.predict(image_data, prompt)
        return result
