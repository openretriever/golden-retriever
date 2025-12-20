import logging

import ray
import requests
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class Blipv2Actor:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        # Initialize the processor and model
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
        )  # doctest: +IGNORE_RESULT

    def predict(self, url, prompt):
        # Load the image
        image = Image.open(requests.get(url, stream=True).raw)
        # Process inputs
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device, dtype=torch.float16
        )
        # Generate text
        # Ensure non-empty input and set appropriate max_length or max_new_tokens
        if inputs.input_ids.nelement() == 0:
            logging.error(
                "Processed inputs are empty. Please check the processor and inputs."
            )
            return "Error: Processed inputs are empty."
        else:
            # Adjust max_length or use max_new_tokens as needed
            generated_ids = self.model.generate(
                **inputs, max_length=100
            )  # Example: set max_length explicitly
            # Alternatively, use max_new_tokens if available in your version of transformers
            # generated_ids = self.model.generate(**inputs, max_new_tokens=50)

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text


if __name__ == "__main__":
    ray.init(num_gpus=1)  # Initialize Ray
    use_gpu = torch.cuda.is_available()

    # Options dictionary for dynamic resource allocation
    actor_options = {"num_gpus": 1} if use_gpu else {}
    # Example usage
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    text = "how many cats are there?"
    prompt = "Question: " + text + " Answer:"

    # Create an instance of the Blipv2Actor
    blipv2_actor = Blipv2Actor.options(**actor_options).remote(use_gpu=use_gpu)

    # Perform prediction
    future = blipv2_actor.predict.remote(url, prompt)
    generated_text = ray.get(future)

    print(f"Generated text: {generated_text}")

    # Shutdown Ray
    ray.shutdown()
