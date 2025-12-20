# Needs latest numpy
# !pip install numpy==1.26.3
# !pip install accelerate>=0.26.0
# !pip install bitsandbytes

import logging
import os
import re

import numpy as np
import ray
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
)

from retriever.models.model_base import LangDetectBase

# Configure logging
logging.basicConfig(level=logging.INFO)


class MolmoBatchClass(LangDetectBase):
    def __init__(
        self, model_name="allenai/Molmo-7B-D-0924", use_gpu=True, use_4bit=True
    ):
        super().__init__(use_gpu)

        # Set device
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        if use_4bit and self.device == "cuda":
            # 4-bit quantization setup
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="fp4",
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_compute_dtype=torch.float32,
                )

                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                )
                logging.info("Using 4-bit quantization")

            except Exception as e:
                logging.warning(
                    f"Failed to configure 4-bit quantization: {e}, falling back to auto dtype"
                )
                self._init_auto_dtype(model_name)
        else:
            # Non-4bit version using auto dtype as per tutorial
            self._init_auto_dtype(model_name)

        if self.device == "cuda":
            logging.info(f"Model loaded on GPU with dtype: {self.model.dtype}")
            logging.info(f"CUDA version: {torch.cuda.get_device_capability()}")
            logging.info(
                f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB"
            )

    def _init_auto_dtype(self, model_name):
        """Initialize model with auto dtype settings from tutorial"""
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype="auto", device_map="auto"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype="auto", device_map="auto"
        )

    def get_device_info(self):
        return f"Model: {next(self.model.parameters()).device}, Input tensors: {self.device}"

    def predict(self, texts, images, points=False, use_scaled_input=True):
        """Process M texts and N images to produce M×N outputs.

        Args:
            texts: Single text string or list of text queries
            images: Single image or list of images
            points: Whether to extract point coordinates from output
            use_scaled_input: Whether to scale images to standard size (576x576) or use original dimensions
        """
        # Convert single inputs to lists
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(texts, str):
            texts = [texts]

        # Convert images to RGB and optionally resize
        target_size = (576, 576)  # Based on the model's expected input size
        processed_images = []
        original_sizes = []  # Keep track of original sizes for point scaling
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Store original size for point scaling
            original_sizes.append(img.size)
            # Resize image if using scaled input
            if use_scaled_input:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            processed_images.append(img)

        all_results = []
        for text in texts:
            # Process batch of images with current text
            inputs = self.processor.process(
                text=text,
                images=processed_images,
            )

            # Move inputs to device and add batch dimension
            inputs = {
                k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()
            }

            # Debug prints for input shapes
            print("\n=== Input Debug Info ===")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: {v.shape}")
            print(f"Number of images: {len(images)}")
            print(f"Text query: {text}")
            print(f"Using scaled input: {use_scaled_input}")

            # Generate with autocast for efficiency
            with torch.autocast(
                device_type=self.device, enabled=True, dtype=torch.bfloat16
            ):
                # TODO: this doesn't work when >1 images?
                output = self.model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
                    tokenizer=self.processor.tokenizer,
                )

            # Debug prints for output
            print("\n=== Output Debug Info ===")
            print(f"Output shape: {output.shape}")
            print(f"Input IDs size: {inputs['input_ids'].size(1)}")

            # Get generated tokens and decode
            generated_tokens = output[0, inputs["input_ids"].size(1) :]
            generated_text = self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            # Debug the generated text
            print("\n=== Generated Text Debug ===")
            print(f"Full text: {generated_text}")
            print("\nTrying to parse points...")

            # Try to extract points for each image
            text_results = []
            for i in range(len(images)):
                if points:
                    # FIXME: we are using same model output for all images
                    # TODO: need to figure out how to separate the outputs for each image
                    if use_scaled_input:
                        # Extract points using target size and scale back
                        point_output = self._extract_points(
                            generated_text, target_size[0], target_size[1]
                        )

                        # Scale points back to original image size
                        orig_w, orig_h = original_sizes[i]
                        scale_w = orig_w / target_size[0]
                        scale_h = orig_h / target_size[1]
                        scaled_points = []
                        for point in point_output:
                            scaled_point = np.array(
                                [point[0] * scale_w, point[1] * scale_h]
                            )
                            scaled_points.append(scaled_point)
                        final_points = scaled_points
                    else:
                        # Extract points using original image size
                        final_points = self._extract_points(
                            generated_text, original_sizes[i][0], original_sizes[i][1]
                        )

                    print(f"\nPoints for image {i}:")
                    print(f"Original size: {original_sizes[i]}")
                    if use_scaled_input:
                        print(f"Target size: {target_size}")
                    print(f"Extracted points: {final_points}")

                    result = {"text": generated_text, "points": final_points}
                else:
                    result = {"text": generated_text}
                text_results.append(result)
            all_results.append(text_results)

        return all_results

    def _extract_points(self, molmo_output, image_w, image_h):
        """Helper method to extract points from model output text"""
        all_points = []
        for match in re.finditer(
            r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"',
            molmo_output,
        ):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                continue

            point = np.array(point)
            if np.max(point) > 100:
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
        return all_points


MolmoBatchActor = ray.remote(MolmoBatchClass)


def draw_points(image, points):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for point in points:
        draw.ellipse(
            [point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5],
            fill="blue",
            outline="white",
        )
    return image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    print(f"Using GPU: {use_gpu}")

    if args.debug:
        print("Running in debug mode, don't use Ray")
        actor = MolmoBatchClass(use_gpu=use_gpu, use_4bit=True)
    else:
        print("Running with Ray")
        ray.init()
        actor_options = {"num_gpus": 2} if use_gpu else {}
        actor = MolmoBatchActor.options(**actor_options).remote(
            use_gpu=use_gpu, use_4bit=True
        )

    logging.info("Actor ready for inference.")

    # Test with multiple images and texts
    images = [
        Image.open("tests/images/test_spot_table_cable_1.jpg"),
        # Image.open("tests/images/test_img_cable_with_knots.jpg"),
        # Image.open(requests.get("https://lh3.googleusercontent.com/p/AF1QipOIEVqiw8NbZgicj-zzdTt6gtTaTRzfKMflQM_F=s1360-w1360-h1020", stream=True).raw)
    ]

    # NOTE: the model can't separate out different objects
    texts = [
        # "Point at the table.",
        # "Point at any knots in the cable."
        # "Point at the cups."
        "Point at the cups or table. Separate out different objects."
        # "Point at the cups and cable and table."
    ]

    # Will return M×N results (M texts × N images)
    if args.debug:
        results = actor.predict(texts, images, points=True)
    else:
        results = ray.get(actor.predict.remote(texts, images, points=True))
    logging.info("Batch inference complete.")

    # Create tmp directory for outputs
    tmp_dir = os.path.join("src", "models", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Process and save results
    for i, text_results in enumerate(results):
        for j, result in enumerate(text_results):
            logging.info(f"\nResults for text '{texts[i]}' with image {j+1}:")
            logging.info(f"VLM Output Text:\n{result['text']}")
            logging.info(f"Detected Points: {result['points']}")

            drawn_image = draw_points(images[j].copy(), result["points"])
            result_path = os.path.join(tmp_dir, f"molmo_points_text{i}_img{j}.png")
            drawn_image.save(result_path)
            logging.info(f"Result image saved to: {result_path}")

    ray.shutdown()
