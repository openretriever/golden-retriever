# Sample script to call a ray model hosted on a different machine from root directory of the project:
# Run from the root (Retreiver/) directory
from glob import glob

import ray
import requests
import torch
from PIL import Image

from retriever.models.detection.groundingdino_actor import GroundingDinoActor

if __name__ == "__main__":
    ignore = glob("*/")
    ignore = [item for item in ignore if "src" not in item]
    ignore = ignore + glob("src/*/")
    ignore = [item for item in ignore if "models" not in item]
    ignore = ["\\" + item for item in ignore]
    ignore.append("\\.git\\")

    # ignore = ['\\assets\\', '\\data\\', '\\docs\\', '\\examples\\', '\\external\\', '\\misc\\', '\\scripts\\', '\\tests\\', '\\src\\actors\\', '\\src\\collectors\\', '\\src\\config\\', '\\src\\dataset\\', '\\src\\envs\\', '\\src\\mappers\\', '\\src\\planners\\', '\\src\\robots\\', '\\src\\skills\\', '\\src\\skill_policy\\', '\\src\\skill_training\\', '\\src\\utils\\', '\\src\\__pycache__\\', '\\.git\\']

    # Start the Ray cluster
    ray.init(
        address="ray://grail-mercury.neu.edu:10001",
        runtime_env={"working_dir": ".", "excludes": ignore},
    )

    try:
        # Create the actor, with or without GPU
        use_gpu = torch.cuda.is_available()
        actor_options = {"num_gpus": 1} if use_gpu else {}

        actor = GroundingDinoActor.options(**actor_options).remote(use_gpu=use_gpu)

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        result = ray.get(actor.predict.remote(image, "a cat. a remote control."))
        print(result)

        ray.shutdown()

    except Exception as e:
        print(e)
        ray.shutdown()
