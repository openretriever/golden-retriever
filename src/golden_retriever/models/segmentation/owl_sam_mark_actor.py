import numpy as np
import ray
import requests
import supervision as sv
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor


# A class integrating OWLv2 detection and SAM segmentation with Supervision visualization
@ray.remote
class OwlSamVisActor:
    def __init__(
        self,
        owlv2_model_name="google/owlv2-base-patch16-ensemble",
        sam_model_name="facebook/sam-vit-huge",
        use_gpu=False,
    ):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Initialize OWLv2 processor and model
        self.owlv2_processor = Owlv2Processor.from_pretrained(owlv2_model_name)
        self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(owlv2_model_name).to(
            self.device
        )

        # Initialize SAM processor and model
        self.sam_processor = SamProcessor.from_pretrained(sam_model_name)
        self.sam_model = SamModel.from_pretrained(sam_model_name).to(self.device)

    def process_image(self, image, texts):
        # Process images with OWLv2
        inputs = self.owlv2_processor(text=texts, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            owlv2_outputs = self.owlv2_model(**inputs)

        # Post-process OWLv2 outputs
        target_size = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.owlv2_processor.post_process_object_detection(
            outputs=owlv2_outputs, target_sizes=target_size, threshold=0.1
        )[0]

        boxes = results["boxes"]

        # Prepare input points for SAM
        input_points = []
        for box in boxes:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            input_points.append([[center_x.item(), center_y.item()]])

        # Convert points to tensor format
        input_points_tensor = torch.tensor(input_points).unsqueeze(0).to(self.device)

        # Process with SAM
        sam_inputs = self.sam_processor(
            image, input_points=input_points_tensor, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)

            print(sam_outputs)

        # Post-process SAM masks
        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )

        # Prepare detections for visualization
        detections = sv.Detections.from_sam(sam_result=sam_outputs)

        return detections, boxes.tolist()

    def visualize(self, image, detections):
        # Annotate masks
        mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX, opacity=0.3
        )

        # Annotate labels
        labels = [str(i) for i in range(len(detections))]
        label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            text_position=sv.Position.CENTER,
            text_scale=1,
            text_color=sv.Color.white(),
            color=sv.Color.black(),
            text_thickness=2,
        )

        # Apply annotations
        annotated_image = mask_annotator.annotate(
            scene=image.copy(), detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        # Display image
        sv.plot_image(annotated_image)


# Example usage
if __name__ == "__main__":
    mode_list = ["local", "cluster", "client"]
    mode = mode_list[0]
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
        ray.init(address="ray://localhost:10001", runtime_env=runtime_env)
        use_gpu = True
    else:
        raise ValueError

    actor_options = {"num_gpus": 1} if use_gpu else {}
    vis_actor = OwlSamVisActor.options(**actor_options).remote(use_gpu=use_gpu)

    # Download an example image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    # Convert image to array format for visualization
    image_np = np.array(image)

    # Process and visualize
    detections, boxes = ray.get(
        vis_actor.process_image.remote(image, ["a photo of a cat", "a photo of a dog"])
    )
    ray.get(vis_actor.visualize.remote(image_np, detections))
