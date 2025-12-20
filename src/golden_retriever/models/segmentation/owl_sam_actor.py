import logging
from typing import List, Tuple, Union

import PIL
import ray
import requests
import torch
from PIL import Image
from rich import pretty, print
from rich.logging import RichHandler
from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor

# from retriever.models.model_base import LangSegBase

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

# Config for rich's pretty printing
pretty.install()


@ray.remote
class OwlSamActor:
    def __init__(
        self,
        sam_model_name: str = "facebook/sam-vit-huge",
        owlv2_model_name: str = "google/owlv2-base-patch16-ensemble",
        use_gpu: bool = False,
    ):
        super().__init__()
        self.device = "cuda" if use_gpu else "cpu"

        # Load OWLv2 model and processor
        self.owlv2_processor = Owlv2Processor.from_pretrained(owlv2_model_name)
        self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(owlv2_model_name).to(
            self.device
        )

        # Load SAM model and processor
        self.sam_processor = SamProcessor.from_pretrained(sam_model_name)
        self.sam_model = SamModel.from_pretrained(sam_model_name).to(self.device)

    def predict(
        self,
        image: PIL.Image.Image,
        texts: Union[str, List[str]],
    ) -> Tuple:
        # Process the image and texts with OWLv2
        inputs = self.owlv2_processor(text=texts, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            owlv2_outputs = self.owlv2_model(**inputs)
        # Post-process the outputs to get the bounding boxes
        target_size = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.owlv2_processor.post_process_object_detection(
            outputs=owlv2_outputs, target_sizes=target_size, threshold=0.3
        )[0]

        # Use the bounding boxes to define input points for SAM
        input_points = []
        boxes = results["boxes"]
        for box in boxes:
            # Calculate the center of each box as input points
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            # Wrap in another list
            input_points.append([[center_x.item(), center_y.item()]])

        # Convert input_points to a tensor with the correct shape
        # NOTE: don't move to CUDA as the processor needs to convert to numpy
        input_points_tensor = torch.tensor(input_points).unsqueeze(0)
        # Shape becomes [1, number of boxes, 1, 2]

        # Process the image and input points with SAM
        sam_inputs = self.sam_processor(
            image, input_points=input_points_tensor, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

        # Post-process SAM masks
        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )

        scores = sam_outputs.iou_scores.cpu().numpy()  # Extracting scores as an example
        return masks, scores, boxes.tolist()


if __name__ == "__main__":
    mode_list = ["local", "cluster", "client"]
    mode = mode_list[2]
    runtime_env = {
        "excludes": ["./src/envs/gibson_example/Eudora.glb"],
        "pip": [
            "rich",
            # "hydra-core",
            # "torch",
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
        ray.init(address="ray://localhost:10001", runtime_env=runtime_env)
        # ray.init(address="ray://128.30.227.158:10001", runtime_env=runtime_env)
        use_gpu = True
    else:
        raise ValueError

    actor_options = {"num_gpus": 1} if use_gpu else {}
    lang_seg_actor = OwlSamActor.options(**actor_options).remote(use_gpu=use_gpu)
    OwlSamActor.options(runtime_env=runtime_env)

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    texts = ["a photo of a cat", "a photo of a dog"]

    # from retriever.models.common_utils import Timer
    # with Timer(enable_print=True) as timer:
    masks, scores, boxes = ray.get(lang_seg_actor.predict.remote(image, texts))

    print("Detected boxes:", boxes)
    print("Segmentation scores:", scores)
    # Further processing can be done here

    ray.shutdown()
