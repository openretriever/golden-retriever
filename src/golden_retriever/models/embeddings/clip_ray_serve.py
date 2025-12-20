import ray
import starlette.requests as requests
from fastapi import FastAPI
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse
from transformers import CLIPModel, CLIPProcessor

from retriever.models.model_base import BaseModelServer

app = FastAPI()


@serve.deployment(
    name="clip_model_service",
    num_replicas=1,
    max_concurrent_queries=8,
    # ray_actor_options={"num_gpus": 0.2},
)
# @serve.ingress(app)
class CLIPModelService(BaseModelServer):
    def __init__(
        self,
        use_gpu: bool = False,
        processor: str = "openai/clip-vit-base-patch32",
        model: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__(use_gpu)
        self.device = "cuda" if use_gpu else "cpu"
        self.processor = CLIPProcessor.from_pretrained(processor)
        self.model = CLIPModel.from_pretrained(model)

    # async def __call__(self, request: requests.Request):
    async def request(self, request: requests.Request):
        # This example assumes you're sending JSON with "texts" and "images" keys
        json = await request.json()
        texts = json["texts"]
        images = json["images"]  # You will need to handle image loading

        # TODO use below
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        outputs = self.model(**inputs)
        return {"logits": outputs.logits_per_image.tolist()}

    async def _predict_single(self, texts, images):
        """Prediction for single run - batching logic outside"""

        # Process texts and images to tensors
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )

        # Move the processed tensors to the same device as the model
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Optionally, move outputs back to CPU if necessary, e.g., outputs = outputs.to('cpu')
        outputs = self.model(**inputs)

        # time.sleep(1)  # TODO test
        # asyncio.sleep(1)

        return {"logits": outputs.logits_per_image.tolist()}

    # TODO experiment
    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def __call__(self, requests, naive_sequencing: bool = True):
        """Entrance, handle dispatching"""

        if naive_sequencing:
            responses = []

            for request in requests:
                _texts = request["texts"]
                _images = request["images"]
                _response = self._predict_single(_texts, _images)
                responses.append(_response)

        else:
            texts = [request["texts"] for request in requests]
            images = [request["images"] for request in requests]

            # FIXME get num of texts and num of images, which will later be used for de-batching
            texts_len = [len(request["texts"]) for request in requests]
            images_len = [len(request["images"]) for request in requests]

            # Preprocess images: Assuming 'images' are file paths or PIL images; adjust as necessary
            processed_images = [
                Image.open(image_path) if isinstance(image_path, str) else image_path
                for image_path in images
            ]

            # Process batch of texts and images
            inputs = self.processor(
                text=texts, images=processed_images, return_tensors="pt", padding=True
            )

            # Perform prediction
            outputs = self.model(**inputs)

            # Extract logits or any other required information from outputs
            logits = (
                outputs.logits_per_image.detach().numpy()
            )  # Assuming you want logits; adjust as needed

            # Convert logits to list of dicts or any other format you need
            # results = [{"logits": logit.tolist()} for logit in logits]
            # TODO de-batching - 1st output = 0:texts_len[0], 2nd output = texts_len[0]:texts_len[1], etc.
            responses = None
            # FIXME input is M texts x N images, if we batch here, then output is (M x B) x (N x B)

        return responses


if __name__ == "__main__":
    # Initialize Ray and Serve
    ray.init()
    serve.start()

    # Predict
    image_path = "./dog_test.jpeg"
    image = Image.open(image_path)

    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of pretty flowers",
        "a photo of a car",
    ]
    images = [image]

    # Start deployment
    clip_deployment = CLIPModelService.bind(model="openai/clip-vit-base-patch32")
    clip_handle: DeploymentHandle = serve.run(clip_deployment)

    print(clip_handle.app_name, clip_handle.deployment_name)

    # It's a special response that we can query using .result() (not ray.get(xxx)!)
    response: DeploymentResponse = clip_handle.remote(
        requests=dict(texts=texts, images=images)
    )
    result = response.result()

    print(response, result)

    # Batching case
    # See: https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html#serve-performance-batching-requests
    responses = [
        clip_handle.remote(requests=dict(texts=texts, images=images)) for _ in range(16)
    ]
    print([response.result() for response in responses])
