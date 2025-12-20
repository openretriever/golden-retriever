# pip install numpy==1.26.3

import logging
import os
from glob import glob

import ray
import requests
import torch
from PIL import Image

from retriever.models.segmentation.sam2_actor import SAM2Actor
from retriever.models.vlms.molmo_quantized_actor import MolmoQuantizedActor, draw_points

if __name__ == "__main__":
    host_local = True

    if not host_local:
        ignore = glob("*/")
        ignore = [item for item in ignore if "src" not in item]
        ignore = ignore + glob("src/*/")
        ignore = [item for item in ignore if "models" not in item]
        ignore = ["\\" + item for item in ignore]
        ignore.append("\\.git\\")

        ray.init(
            address="ray://ray-host:10001",
            runtime_env={"working_dir": ".", "excludes": ignore},
        )
    else:
        ray.init()

    print(ray.available_resources())

    # Adjust based on your setup
    use_gpu = torch.cuda.is_available()

    sam_actor_options = {"num_gpus": 1} if use_gpu else {}
    molmo_actor_options = {"num_gpus": 1} if use_gpu else {}

    molmo_actor = MolmoQuantizedActor.options(**molmo_actor_options).remote(
        use_gpu=use_gpu
    )
    sam_actor = SAM2Actor.options(**sam_actor_options).remote(use_gpu=use_gpu)

    logging.info("Actor ready for inference.")
    url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"  # noqa
    image = Image.open(requests.get(url, stream=True).raw)

    # Create tmp directory for outputs
    tmp_dir = os.path.join("src", "models", "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    result = ray.get(
        molmo_actor.predict.remote(
            image, "Point at the front wheel of the car.", points=True
        )
    )

    drawn_image = draw_points(image, result["points"])
    # Save the intermediate result
    drawn_path = os.path.join(tmp_dir, "points.png")
    drawn_image.save(drawn_path)
    print(f"Points visualization saved to: {drawn_path}")

    result = ray.get(
        sam_actor.predict.remote(drawn_image, input=result["points"], type="point")
    )
    # Save the final result
    result_path = os.path.join(tmp_dir, "segmentation.png")
    result.save(result_path)
    print(f"Segmentation result saved to: {result_path}")

    logging.info("Inference complete. Check the output files in ./src/models/tmp/")
