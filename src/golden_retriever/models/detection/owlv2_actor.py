import logging

import ray
import requests
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from retriever.models.model_base import LangDetectBase

# Configure logging
logging.basicConfig(level=logging.INFO)


@ray.remote
class OWLV2Actor(LangDetectBase):
    def __init__(self, model_name="google/owlv2-base-patch16-ensemble", use_gpu=False):
        super().__init__(use_gpu)
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
        outputs = self.model(**inputs)

        # Convert outputs (bounding boxes and class logits) to desired format
        target_sizes = torch.Tensor([image.size[::-1] for image in images]).to(
            self.device
        )
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.1
        )

        # Simplified output for demonstration. Customize as needed.
        simplified_results = []
        for i, result in enumerate(results):
            text_queries = texts[i]
            boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
            detections = []
            for box, score, label in zip(boxes, scores, labels, strict=False):
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


if __name__ == "__main__":
    ray.init()
    # ray.init("ray://localhost:10002")
    print(ray.available_resources())

    # Adjust based on your setup
    use_gpu = torch.cuda.is_available()

    actor_options = {"num_gpus": 1} if use_gpu else {}

    # Create an actor
    owlv2_actor = OWLV2Actor.options(**actor_options).remote(use_gpu=use_gpu)

    # Example usage
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # noqa
    image = Image.open(requests.get(url, stream=True).raw)

    texts = [["a photo of a cat", "a photo of a dog"]]
    results = ray.get(owlv2_actor.predict.remote([image], texts))

    print(results)

    ray.shutdown()
