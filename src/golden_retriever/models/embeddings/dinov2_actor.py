import logging

import ray
import requests
import torch
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class DinoV2Actor:
    def __init__(self, model_name="facebook/dinov2-base", use_gpu=False):
        from transformers import AutoImageProcessor, AutoModel

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Trace model
        # self.traced_model = torch.jit.trace(self.model, [inputs.pixel_values])

        # Ensure return_dict is False for compatibility with torch.jit.trace
        self.model.config.return_dict = False

    def predict_and_trace(self, image_pil):
        # Process image
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_states = outputs[0]
        # Trace model
        traced_model = torch.jit.trace(self.model, [inputs.pixel_values])
        traced_outputs = traced_model(inputs.pixel_values)

        max_diff = (last_hidden_states - traced_outputs[0]).abs().max()
        # Calculate maximum absolute difference
        # max_diff = (last_hidden_states - traced_outputs).abs().max().item()

        return traced_outputs[0]

    def predict(self, image_pil):
        # Process image
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_states = outputs[0]

        return outputs, last_hidden_states


if __name__ == "__main__":
    ray.init(num_gpus=1)  # Add arguments as necessary, e.g., address, num_gpus

    # Variable to control GPU usage
    use_gpu = torch.cuda.is_available()

    # URL of the image to process
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # Load image from URL
    image = requests.get(url, stream=True).raw
    # Load image with PIL
    image_pil = Image.open(image)

    # Options dictionary for dynamic resource allocation
    actor_options = {"num_gpus": 1} if use_gpu else {}

    # Create an actor instance with dynamic GPU allocation
    dinov2_actor = DinoV2Actor.options(**actor_options).remote(use_gpu=use_gpu)

    # Perform prediction and tracing
    max_diff_future = dinov2_actor.predict_and_trace.remote(image_pil)

    max_diff = ray.get(max_diff_future)
    print(f"Maximum absolute difference, shape: {max_diff.shape}")
    print(f"Maximum absolute difference: {max_diff}")

    import matplotlib.pyplot as plt

    plt.imshow(max_diff.detach().cpu().permute(1, 2, 0).numpy())
    plt.show()

    # Prediction output
    features_future = dinov2_actor.predict.remote(image_pil)
    outputs, features = ray.get(features_future)
    print(f"Features shape: {features.shape}")

    # Shutdown Ray
    ray.shutdown()
