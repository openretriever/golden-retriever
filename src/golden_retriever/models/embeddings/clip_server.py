from typing import List

import clip
import ray
from PIL import Image

from retriever.config.config_sacred_test import get_clip_model, get_device
from retriever.models.model_base import ModelServer


@ray.remote(num_cpus=1, num_gpus=0.1)
class ClipEncoder(ModelServer):
    def __init__(self, model):
        self.device = get_device()
        self.model, preprocess = clip.load(get_clip_model(), device=self.device)

    def encode_text(self, text: List[str]):
        return self.model.encode_text(clip.tokenize(text).to(self.device))

    def encode_image(self, image: Image):
        return self.model.encode_image(image)
