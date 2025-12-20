import pathlib

import ray
import torch
from PIL import Image

from retriever.models.segmentation.owl_sam_actor import OwlSamActor

# Variable to control GPU usage
# Set to True to use GPU, False to not use GPU
use_gpu = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {num_gpus}")

# Note: Replace with your Ray cluster's address
ray.init(num_gpus=0 if not use_gpu else num_gpus)
print(ray.available_resources())

# Options dictionary to dynamically set num_gpus
actor_options = {"num_gpus": 0.1} if use_gpu else {}

# Create an actor instance with dynamic GPU allocation
actor_options = {"num_gpus": 1} if use_gpu else {}
lang_seg_actor = OwlSamActor.options(**actor_options).remote(use_gpu=use_gpu)

image1 = pathlib.Path("./spot_rgb_test1.jpg")
image = Image.open(image1).convert("RGB")
texts = ["a photo of a cat", "a photo of a dog"]

# TODO use
