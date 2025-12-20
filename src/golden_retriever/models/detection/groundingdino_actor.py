import logging

import ray
import requests
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from retriever.models.model_base import LangDetectBase

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class GroundingDinoActor(LangDetectBase):
    def __init__(self, model_name="IDEA-Research/grounding-dino-tiny", use_gpu=False):
        super().__init__(use_gpu)
        self.device = "cuda" if use_gpu else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(
            self.device
        )

    def predict(self, image, text):
        """Assuming `image` is a single PIL.Image objects and `text` is a single string"""

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        box_threshold = 0.3
        text_threshold = 0.3

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )
        texts = text.split(".")
        # Simplified output for demonstration. Customize as needed.
        """
        simplified_results = []
        for i, result in enumerate(results):
            text_queries = texts[i]
            boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
            detections = []
            for box, score, label in zip(boxes, scores, labels):
                box = [round(b, 2) for b in box.tolist()]
                detections.append(
                    {
                        "text": text_queries[label],
                        "confidence": round(score.item(), 3),
                        "location": box,
                    }
                )
            simplified_results.append(detections)
        return simplified_results
        """
        return results


if __name__ == "__main__":
    # Start a Ray cluster
    ray.init()

    print(ray.available_resources())

    # Adjust based on your setup
    use_gpu = torch.cuda.is_available()

    actor_options = {"num_gpus": 1} if use_gpu else {}

    actor = GroundingDinoActor.options(**actor_options).remote(use_gpu=use_gpu)
    logging.info("Actor ready for inference.")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # noqa
    image = Image.open(requests.get(url, stream=True).raw)

    result = ray.get(actor.predict.remote(image, "a cat. a remote control."))
    logging.info("Inference complete.")

    print(result)

    ray.shutdown()
