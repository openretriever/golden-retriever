"""
Retriever Example: Rerun Visualization with Webcam & Perception

This example demonstrates how to:
1. Capture images from a webcam.
2. Run open-vocabulary object detection (OwlViT) and segmentation (SAM).
3. Log all data (images, bboxes, masks) to Rerun for real-time visualization.

Dependencies:
    pixi run demo-webcam-rerun

Usage:
    # Just run it!
    pixi run demo-webcam-rerun
"""

import argparse
import time
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import cv2
import torch
import rerun as rr
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

from transformers import Owlv2Processor, Owlv2ForObjectDetection, SamModel, SamProcessor

from retriever.flow import Flow, flow_io, Rate, Pipeline

# ==============================================================================
# Data Structures
# ==============================================================================

@flow_io
@dataclass
class ImageMsg:
    frame: np.ndarray  # BGR
    timestamp: float

@flow_io
@dataclass
class PerceptionResult:
    image: np.ndarray  # BGR
    timestamp: float
    boxes: List[List[float]]  # [x1, y1, x2, y2]
    scores: List[float]
    labels: List[str]
    masks: Optional[np.ndarray] = None  # [N, H, W] boolean masks

# ==============================================================================
# Nodes / Flows
# ==============================================================================

class WebcamSource(Flow[None, ImageMsg]):
    def __init__(self, device_index=0, width=640, height=480):
        super().__init__()
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap = None

    def init(self):
        print(f"[Webcam] Opening camera {self.device_index}...")
        self.cap = cv2.VideoCapture(self.device_index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        else:
            print(f"[Webcam] Failed to open camera {self.device_index}. Using mock.")
            self.cap = None

    def run(self, _) -> ImageMsg:
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ImageMsg(frame=frame, timestamp=time.time())
        
        # Mock fallback
        time.sleep(0.03)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Bouncing box
        t = time.time()
        x = int((t * 100) % (self.width - 50))
        y = int((t * 80) % (self.height - 50))
        cv2.rectangle(img, (x, y), (x+50, y+50), (0, 255, 0), -1)
        return ImageMsg(frame=img, timestamp=time.time())

    def init_config(self):
        return {
            "device_index": self.device_index,
            "width": self.width,
            "height": self.height
        }

class OwlSamModel:
    """Helper class to load models once."""
    def __init__(self, device="cpu"):
        self.device = device
        print(f"[Model] Loading OwlViT & SAM on {device}...")
        
        # Load OwlViT
        self.owl_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
        
        # Load SAM
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        print("[Model] Loaded.")

    def predict(self, image_bgr: np.ndarray, text_queries: List[str], score_threshold=0.1) -> PerceptionResult:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # 1. Detection (OwlViT)
        inputs = self.owl_processor(text=[text_queries], images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.owl_model(**inputs)
            
        target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)
        results = self.owl_processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=score_threshold
        )[0]
        
        boxes = results["boxes"]
        scores = results["scores"]
        labels_idx = results["labels"]
        
        detected_boxes = []
        detected_scores = []
        detected_labels = []
        
        input_points = []
        
        for box, score, label_idx in zip(boxes, scores, labels_idx):
            box_lst = box.tolist() # [x1, y1, x2, y2]
            detected_boxes.append(box_lst)
            detected_scores.append(score.item())
            detected_labels.append(text_queries[label_idx])
            
            # Center point for SAM
            cx = (box_lst[0] + box_lst[2]) / 2
            cy = (box_lst[1] + box_lst[3]) / 2
            input_points.append([[cx, cy]])

        masks_np = None
        if len(input_points) > 0:
            # 2. Segmentation (SAM)
            # Reshape input points: [batch, point_batch, num_points_per_mask, 2]
            # We treat each box as a separate prompt.
            # Actually SamProcessor expects: input_points (batch_size, num_points, 2)
            
            # Simple approach: Standard SAM batch processing can be tricky with varying points.
            # Using the processor helper:
            input_points_tensor = torch.tensor(input_points).to(self.device) # [N, 1, 2]
            
            # We need to replicate image for each box if we batch this way, OR usage simpler API.
            # Let's use the processor correctly. It handles batching commands.
            # But here we have 1 image and N points.
            
            # Hack for now: Just one detailed call per image? 
            # SAM processor logic in `OwlSamActor` was:
            # input_points_tensor = [1, number_of_boxes, 1, 2]
            # sam_inputs = processor(image, input_points=...)
            
            input_points_tensor = torch.tensor(input_points, dtype=torch.float32).unsqueeze(0) # [1, N, 1, 2]
            
            sam_inputs = self.sam_processor(
                pil_image, input_points=input_points_tensor, return_tensors="pt"
            )
            # Sanitize for MPS (no float64)
            for k, v in sam_inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.float64:
                        v = v.to(torch.float32)
                    sam_inputs[k] = v.to(self.device)
            
            with torch.no_grad():
                sam_outputs = self.sam_model(**sam_inputs)
            
            # Post process
            # masks shape: [batch, num_boxes, 3 (multimask), H, W]
            sam_masks = self.sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu()
            )
            # sam_masks[0] is [N, 3, H, W]
            # We usually take the best score mask or the first one. Let's take index 0.
            if len(sam_masks) > 0:
                masks_np = sam_masks[0][:, 0, :, :].numpy() # [N, H, W]

        return PerceptionResult(
            image=image_bgr,
            timestamp=time.time(),
            boxes=detected_boxes,
            scores=detected_scores,
            labels=detected_labels,
            masks=masks_np
        )

class PerceptionFlow(Flow[ImageMsg, PerceptionResult]):
    def __init__(self, queries: List[str]):
        super().__init__()
        self.queries = queries
        self.model = None

    def init(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Check for MPS
        if torch.backends.mps.is_available():
            device = "mps"
        
        self.model = OwlSamModel(device=device)

    def run(self, input: ImageMsg) -> PerceptionResult:
        if input is None or input.frame is None or input.frame.size == 0:
            return None
        return self.model.predict(input.frame, self.queries)
    
    def init_config(self):
        return {"queries": self.queries}

class RerunLogFlow(Flow[PerceptionResult, None]):
    def __init__(self):
        super().__init__()
        pass
    
    def init(self):
        rr.init("retriever_perception_demo")
        rr.connect()

    def run(self, res: PerceptionResult):
        if res is None: 
            return
        
        # Debug
        # print(f"[Rerun] Received res: {type(res)}")
        if res.image is None or res.image.size == 0:
            if not getattr(self, "_warned_empty", False):
                print("[Rerun] Warning: Received empty image in result (waiting for model init...)")
                self._warned_empty = True
            return
        
        # Log Image
        # Rerun expects RGB
        img_rgb = cv2.cvtColor(res.image, cv2.COLOR_BGR2RGB)
        rr.set_time_seconds("stable_time", res.timestamp)
        
        rr.log("camera/image", rr.Image(img_rgb))
        
        if res.boxes:
            rr.log(
                "camera/image/detections",
                rr.Boxes2D(
                    array=res.boxes,
                    array_format=rr.Box2DFormat.XYXY,
                    labels=[f"{l} {s:.2f}" for l, s in zip(res.labels, res.scores)]
                )
            )
            
        if res.masks is not None:
             # Overlay masks
             # We can log them as Annotations or just generic masks
             # Creating a single segmentation image or multiple?
             # Rerun supports mask logging. 
             # Let's combine them into a single segmentation image for simplicity?
             # Or log each as a separate entity? "camera/image/mask/{i}"
             
             # Combined mask
             combined_mask = np.zeros(res.image.shape[:2], dtype=np.uint8)
             for i, mask in enumerate(res.masks):
                 combined_mask[mask] = i + 1
             
             rr.log("camera/image/segmentation", rr.SegmentationImage(combined_mask))


# ==============================================================================
# Main
# ==============================================================================

def cleanup_checkpoints():
    print("[Cleanup] Cleaning up model checkpoints...")
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    if not cache_dir.exists():
        return

    targets = [
        "models--google--owlv2-base-patch16-ensemble",
        "models--facebook--sam-vit-base"
    ]

    for item in cache_dir.iterdir():
        if item.is_dir() and any(t in item.name for t in targets):
            print(f"[Cleanup] Removing {item}")
            shutil.rmtree(item)
    print("[Cleanup] Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="dora", choices=["dora", "multiprocessing"])
    parser.add_argument("--queries", default="person,face,cell phone", help="Comma separated queries")
    parser.add_argument("--cleanup", action="store_true", help="Delete model checkpoints after run to save space")
    args = parser.parse_args()

    queries = [q.strip() for q in args.queries.split(",")]

    # Initialize Rerun
    rr.init("retriever_perception_demo", spawn=True)

    # Build Pipeline
    p = Pipeline("rerun_demo")
    
    with p:
        cam = WebcamSource() @ Rate(hz=10)
        perc = PerceptionFlow(queries=queries) @ Rate(hz=5)
        logger = RerunLogFlow() @ Rate(hz=10)
        
        cam >> perc >> logger

    if args.cleanup:
        import atexit
        atexit.register(cleanup_checkpoints)

    print("Pipeline starting. Check Rerun window!")
    try:
        p.run(backend=args.backend)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
