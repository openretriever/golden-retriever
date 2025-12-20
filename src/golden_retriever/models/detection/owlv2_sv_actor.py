import logging

import ray
import requests
import supervision as sv
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class OWLV2Actor:
    def __init__(self, model_name="google/owlv2-base-patch16-ensemble", use_gpu=False):
        self.device = "cuda" if use_gpu else "cpu"
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)

    def predict(self, images, texts):
        """Assuming `images` is a list of PIL.Image objects and `texts` is a list of text queries"""
        if not isinstance(images, list):
            images = [images]

        inputs = self.processor(text=texts, images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs (bounding boxes and class logits) to desired format
        target_sizes = torch.Tensor([image.size[::-1] for image in images]).to(
            self.device
        )
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.1
        )

        # results_aggregated = {
        #     # "boxes": np.stack([result["boxes"].cpu().numpy() for result in results]),
        #     # "labels": np.stack([result["labels"].cpu().numpy() for result in results]),
        #     # "scores": np.stack([result["scores"].cpu().numpy() for result in results]),
        #     "boxes": torch.stack([result["boxes"] for result in results]),
        #     "labels": torch.stack([result["labels"] for result in results]),
        #     "scores": torch.stack([result["scores"] for result in results]),
        # }

        # Convert to Detections objects
        detections_list = sv.Detections.from_transformers(results[0])
        # detections_list = sv.Detections.from_transformers(results_aggregated)

        print(detections_list)
        # TODO check after this

        # For demonstration, return simplified detection results
        simplified_results = []
        for i, detections in enumerate(detections_list):
            detection_data = [
                {
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "location": [round(coord, 2) for coord in xyxy],
                }
                for class_id, confidence, xyxy in zip(
                    detections.class_id,
                    detections.confidence,
                    detections.xyxy,
                    strict=False,
                )
            ]
            simplified_results.append(detection_data)

        return simplified_results


if __name__ == "__main__":
    ray.init(local_mode=True)
    print(ray.available_resources())

    use_gpu = torch.cuda.is_available()
    actor_options = {"num_gpus": 1} if use_gpu else {}

    owlv2_actor = OWLV2Actor.options(**actor_options).remote(use_gpu=use_gpu)

    # Example usage
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    texts = [["a photo of a cat", "a photo of a dog"]]
    results = ray.get(owlv2_actor.predict.remote([image], texts))

    print(results)

    ray.shutdown()
