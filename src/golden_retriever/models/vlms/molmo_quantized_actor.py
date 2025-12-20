# Needs latest numpy
# !pip install numpy==1.26.3
# !pip install accelerate>=0.26.0
# !pip install bitsandbytes

import logging
import os
import re
from typing import List

import numpy as np
import ray
import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
)

from retriever.models.model_base import LangDetectBase

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class MolmoQuantizedActor(LangDetectBase):
    def __init__(self, model_name="allenai/Molmo-7B-D-0924", use_gpu=True):
        super().__init__(use_gpu)

        # Set device and dtype
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        arguments = {
            "device_map": "auto",
            "torch_dtype": torch.float32,  # Always use float32 for initialization
            "trust_remote_code": True,
        }

        if self.device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="fp4",
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_compute_dtype=torch.float32,  # Changed from float16
                )
                arguments["quantization_config"] = quantization_config
                logging.info("Using 4-bit quantization")
            except Exception as e:
                logging.warning(f"Failed to configure quantization: {e}")

        self.processor = AutoProcessor.from_pretrained(model_name, **arguments)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **arguments)

        if self.device == "cuda":
            logging.info(f"Model loaded on GPU with dtype: {self.model.dtype}")
            logging.info(
                f"CUDA version: {torch.cuda.get_device_capability()}"
            )  # Changed this line
            logging.info(
                f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB"
            )

    def get_device_info(self):
        return f"Model: {next(self.model.parameters()).device}, Input tensors: {self.device}"

    def predict(self, image, text, points=False):
        """Assuming `image` is a single PIL.Image objects and `text` is a single string"""

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor.process(
            images=[image],
            text=text,
        )

        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer,
        )

        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        if points:

            def extract_points(molmo_output, image_w, image_h):
                all_points = []
                for match in re.finditer(
                    r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"',
                    molmo_output,
                ):
                    try:
                        point = [float(match.group(i)) for i in range(1, 3)]
                    except ValueError:
                        pass
                    else:
                        point = np.array(point)
                        if np.max(point) > 100:
                            # Treat as an invalid output
                            continue
                        point /= 100.0
                        point = point * np.array([image_w, image_h])
                        all_points.append(point)
                return all_points

            point_output = extract_points(generated_text, image.size[0], image.size[1])
            results = {"text": generated_text, "points": point_output}
        else:
            results = {"text": generated_text}

        return results


def draw_points(
    image: Image.Image, points: List[List[float]], color: str = "red"
) -> Image.Image:
    """Draw points on image with size proportional to image dimensions"""
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Scale point radius and outline width with image size
    point_radius = max(5, min(width, height) // 100)  # Minimum 5px radius
    outline_width = max(2, point_radius // 2)  # Minimum 2px outline

    for x, y in points:
        # Draw point with outline
        bbox = [x - point_radius, y - point_radius, x + point_radius, y + point_radius]
        draw.ellipse(bbox, fill=color, outline="white", width=outline_width)

    return image


if __name__ == "__main__":
    # Start a Ray cluster
    ray.init()

    print(ray.available_resources())

    # Adjust based on your setup
    use_gpu = torch.cuda.is_available()

    actor_options = {"num_gpus": 1} if use_gpu else {}

    actor = MolmoQuantizedActor.options(**actor_options).remote(use_gpu=use_gpu)
    logging.info("Actor ready for inference.")

    # url = "https://lh3.googleusercontent.com/p/AF1QipOIEVqiw8NbZgicj-zzdTt6gtTaTRzfKMflQM_F=s1360-w1360-h1020"
    # image = Image.open(requests.get(url, stream=True).raw)

    # # result = ray.get(actor.predict.remote(image, "Describe this image."))
    # result = ray.get(
    #     actor.predict.remote(image, "Point at the handle of the door.", points=True)
    # )

    image = Image.open("tests/images/test_spot_table_cable_1.jpg")
    result = ray.get(
        actor.predict.remote(
            image, "Point at the cups and cable and table.", points=True
        )
    )

    logging.info("Inference complete.")

    # Log the VLM output text
    logging.info("VLM Output Text:\n%s", result["text"])

    # Log the detected points
    logging.info("Detected Points: %s", result["points"])

    drawn_image = draw_points(image, result["points"])

    # Create tmp directory for outputs
    tmp_dir = os.path.join("src", "models", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Save the result
    result_path = os.path.join(tmp_dir, "molmo_points.png")
    drawn_image.save(result_path)
    logging.info("Result image saved to: %s", result_path)

    ray.shutdown()
