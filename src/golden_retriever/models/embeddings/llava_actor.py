import logging
from typing import List, Union

import PIL
import ray
import requests
import torch
from PIL import Image
from rich import print
from rich.logging import RichHandler
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])


@ray.remote
class LlavaActor:
    def __init__(
        self, model_name: str = "llava-hf/llava-1.5-7b-hf", use_gpu: bool = False
    ):
        self.device = "cuda" if use_gpu else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name)

    def generate_response(
        self,
        images: Union[PIL.Image.Image, List[PIL.Image.Image]],
        texts: Union[str, List[str]],
    ):
        # Prepare inputs
        inputs = self.processor(text=texts, images=images, return_tensors="pt")

        # Generate response
        generate_ids = self.model.generate(**inputs, max_length=30)
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response


if __name__ == "__main__":
    # Initialize Ray
    use_gpu = torch.cuda.is_available()
    ray.init(num_gpus=torch.cuda.device_count() if use_gpu else None)

    # Options dictionary for dynamic resource allocation
    actor_options = {"num_gpus": 1} if use_gpu else {}
    # Example usage
    text = "What's the content of the image?"
    prompt = "<image>\nUSER: " + text + "\nASSISTANT:"
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Create an actor instance
    llava_actor = LlavaActor.options(**actor_options).remote(use_gpu=use_gpu)

    # Perform generation
    response_future = llava_actor.generate_response.remote([image], prompt)
    response = ray.get(response_future)

    print(f"Generated Response: {response}")

    # Shutdown Ray
    ray.shutdown()
