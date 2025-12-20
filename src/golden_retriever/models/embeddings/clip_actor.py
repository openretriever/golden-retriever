"""
Usage:
python -m src.models.embeddings.clip_actor
"""

import logging

import ray
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from retriever.models.model_base import ModelActorBase

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class CLIPActor(ModelActorBase):
    def __init__(self, use_gpu=False):
        super().__init__(use_gpu)
        self.device = "cuda" if use_gpu else "cpu"
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )

    def predict(self, texts, images):
        # Process texts and images to tensors
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )

        # Move the processed tensors to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Perform inference
        outputs = self.model(**inputs)

        # Optionally, move outputs back to CPU if necessary, e.g., outputs = outputs.to('cpu')
        return (
            outputs.logits_per_image.detach().to("cpu").numpy()
        )  # Example, adjust based on your needs


if __name__ == "__main__":
    # Initialize Ray
    # TODO Test: Replace with your Ray cluster's address
    ray.init(num_gpus=1)
    print(ray.available_resources())

    # Variable to control GPU usage
    # use_gpu = True  # Set to True to use GPU, False to not use GPU
    use_gpu = False  # Set to True to use GPU, False to not use GPU

    # Options dictionary to dynamically set num_gpus
    actor_options = {"num_gpus": 1} if use_gpu else {}

    # Create an actor instance with dynamic GPU allocation
    clip_actor = CLIPActor.options(**actor_options).remote(use_gpu=use_gpu)

    print(clip_actor)

    # Predict
    image1_path = "./tests/images/test_img_cable_with_knots.jpg"
    image1 = Image.open(image1_path)
    image2_path = "./tests/images/test_spot_table_cable_1.jpg"
    image2 = Image.open(image2_path)

    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of pretty flowers",
        "a photo of a car",
        "a photo of a cable with knots",
        "a photo of a table",
    ]
    images = [
        image1,
        image2,
    ]

    logits = ray.get(clip_actor.predict.remote(texts, images))

    print(logits)

    ray.shutdown()
