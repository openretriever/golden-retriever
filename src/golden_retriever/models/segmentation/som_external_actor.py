import logging

import ray
import torch

from retriever.models import utils
from retriever.utils.som import som_foractor

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class SomActor:
    def __init__(self, use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    def predict(
        self, image, slider, mode, alpha, label_mode, anno_mode, *args, **kwargs
    ):
        # Load image from URL

        return som_foractor.inference(
            image, slider, mode, alpha, label_mode, anno_mode, *args, **kwargs
        )


if __name__ == "__main__":
    mode_list = ["local", "cluster", "client"]
    mode = mode_list[2]
    runtime_env = {
        "excludes": ["./src/envs/gibson_example/Eudora.glb"],
        "pip": [
            "rich",
            "hydra-core",
            "torch",
            "transformers",
        ],
    }

    if mode == "local":
        ray.init()
        use_gpu = torch.cuda.is_available()
    elif mode == "cluster":
        ray.init(address="auto", runtime_env=runtime_env)
        use_gpu = True
    elif mode == "client":
        # ray.init(address="ray://localhost:10001", runtime_env=runtime_env)
        # ray.init(address="ray://128.30.227.158:10001", runtime_env=runtime_env)
        ray.init(address="ray://10.188.62.113:10001", runtime_env=runtime_env)
        use_gpu = True

    image = "https://static01.nyt.com/images/2020/09/08/well/physed-cycle-walk/physed-cycle-walk-videoSixteenByNineJumbo1600-v2.jpg"
    image_pil = utils.load_image(image)
    image = image_pil

    slider = 1  # [1-3] # info="Choose in [1, 1.5), [1.5, 2.5), [2.5, 3] for [seem, semantic-sam (multi-level), sam]"
    mode = "Automatic"
    anno_mode = [
        "Mark",
        "Mask",
        "Box",
    ]  # choices=["Mark", "Mask", "Box"], value=['Mark']

    slider_alpha = 0  # info="Choose in [0, 1]"
    label_mode = (
        "Number"  # gr.Radio(['Number', 'Alphabet'], value='Number', label="Mark Mode")
    )
    # image_out = som_foractor.inference(image, slider, mode, slider_alpha, label_mode, anno_mode)

    # Options dictionary for dynamic resource allocation
    actor_options = {"num_gpus": 1} if use_gpu else {}

    # Create an actor instance with dynamic GPU allocation
    som_actor = SomActor.options(**actor_options).remote(use_gpu=use_gpu)

    image_out = ray.get(
        som_actor.predict.remote(
            image, slider, mode, slider_alpha, label_mode, anno_mode
        )
    )
    utils.display_image(image_out[0])
    # Shutdown Ray
    ray.shutdown()
