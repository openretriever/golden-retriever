"""
Retriever Advanced Example: Interactive Vision-VLA Demo

This demo performs real-time open-vocabulary object detection using OWL-ViT.
Users can input natural language prompts (e.g., "detect the coffee mug") via 
a web interface to update the detection targets dynamcially.

How to run:
    pixi run demo-vision
"""
import os
import sys
import time
import queue
import threading
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Ensure the project root and src/ are in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
src_root = os.path.join(project_root, "src")
for path in [src_root, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from retriever.flow import (
    Pipeline, Rate, Flow, flow_io, Events
)

# =============================================================================
# 1. Models & Utilities
# =============================================================================

class VisionDetector:
    """Wrapper for OWL-ViT open-vocabulary detection."""
    def __init__(self, model_id="google/owlvit-base-patch32", device=None):
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[Detector] Loading model {model_id} on {self.device}...")
        self.processor = OwlViTProcessor.from_pretrained(model_id)
        self.model = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print("[Detector] Model loaded.")

    @torch.no_grad()
    def detect(self, image: np.ndarray, queries: List[str], threshold=0.1):
        if not queries:
            return []

        # Convert OpenCV BGR to RGB PIL image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = self.processor(text=[queries], images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)

        i = 0  # Only one image in batch
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            detections.append({
                "box": box, # [xmin, ymin, xmax, ymax]
                "score": round(score.item(), 3),
                "label": queries[label.item()]
            })
        return detections

# =============================================================================
# 2. Flows
# =============================================================================

@flow_io
class WebcamFlow(Flow):
    """Source flow that captures frames from the webcam."""
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam {camera_id}")
        
    def run(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

@flow_io
class InteractionFlow(Flow):
    """Source flow for the interactive detection prompt."""
    def __init__(self, initial_prompt="person, coffee mug, cell phone"):
        self.current_prompt = initial_prompt
        self._prompt_queue = queue.Queue()

    def set_prompt(self, prompt: str):
        self._prompt_queue.put(prompt)

    def run(self):
        try:
            # Non-blocking check for new prompt
            self.current_prompt = self._prompt_queue.get_nowait()
            print(f"[Interaction] Prompt updated to: '{self.current_prompt}'")
        except queue.Empty:
            pass
        return self.current_prompt

@flow_io
class VisionDetectorFlow(Flow):
    """Transform flow that performs object detection."""
    def __init__(self, threshold=0.1):
        self.detector = VisionDetector()
        self.threshold = threshold

    def run(self, image, prompt):
        start_time = time.time()
        queries = [q.strip() for q in prompt.split(",")]
        detections = self.detector.detect(image, queries, threshold=self.threshold)
        
        latency = (time.time() - start_time) * 1000
        return {
            "image": image,
            "detections": detections,
            "latency_ms": latency,
            "prompt": prompt
        }

@flow_io
class WebStreamingSink(Flow):
    """Sink flow that hosts a FastAPI server for streaming and interaction."""
    def __init__(self, host="0.0.0.0", port=8000, interaction_flow=None):
        self.host = host
        self.port = port
        self.interaction_flow = interaction_flow
        
        self.latest_frame = None
        self.latest_data = {}
        self._frame_lock = threading.Lock()
        
        # FastAPI setup
        self.app = FastAPI(title="Retriever Vision-VLA Demo")
        self.setup_routes()
        
        # Start server in thread
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

    def setup_routes(self):
        # Statics & Templates
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        self.templates = Jinja2Templates(directory=static_dir)

        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})

        @self.app.get("/video_feed")
        async def video_feed():
            return StreamingResponse(
                self.gen_frames(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

        @self.app.post("/update_prompt")
        async def update_prompt(data: Dict[str, str]):
            prompt = data.get("prompt", "")
            if self.interaction_flow:
                self.interaction_flow.set_prompt(prompt)
                return {"status": "success", "new_prompt": prompt}
            return {"status": "error", "message": "InteractionFlow not linked"}

        @self.app.get("/stats")
        async def get_stats():
            with self._frame_lock:
                return {
                    "latency_ms": round(self.latest_data.get("latency_ms", 0), 1),
                    "prompt": self.latest_data.get("prompt", ""),
                    "detections": len(self.latest_data.get("detections", []))
                }

    def gen_frames(self):
        while True:
            with self._frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.01)
                    continue
                
                # Annotate frame
                frame = self.latest_frame.copy()
                detections = self.latest_data.get("detections", [])
                
                for det in detections:
                    box = det["box"]
                    label = det["label"]
                    score = det["score"]
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {score}", (int(box[0]), int(box[1]-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03) # ~30 FPS UI refresh

    def _run_server(self):
        log_config = uvicorn.config.LOGGING_CONFIG
        log_config["loggers"]["uvicorn"]["level"] = "WARNING"
        uvicorn.run(self.app, host=self.host, port=self.port, log_config=log_config)

    def run(self, data):
        with self._frame_lock:
            self.latest_frame = data["image"]
            self.latest_data = {
                "detections": data["detections"],
                "latency_ms": data["latency_ms"],
                "prompt": data["prompt"]
            }

# =============================================================================
# 3. Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Interactive Vision-VLA Demo")
    parser.add_argument("--backend", type=str, default="multiprocessing", 
                        choices=["multiprocessing", "dora"], help="Runtime backend")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("Retriever Advanced Example: Interactive Vision-VLA")
    print("="*60 + "\n")

    # Build Pipeline
    p = Pipeline("vision_vla_interactive")
    
    # Sources
    camera = p.add_flow(WebcamFlow(), name="camera", clock=Rate(hz=15))
    interaction = p.add_flow(InteractionFlow(), name="interaction", clock=Rate(hz=5))
    
    # Transform
    detector = p.add_flow(VisionDetectorFlow(), name="detector")
    p.connect(camera, detector.image)
    p.connect(interaction, detector.prompt)
    
    # Sink
    web_sink = WebStreamingSink(interaction_flow=interaction.flow_instance)
    p.add_flow(web_sink, name="web_server")
    p.connect(detector, "web_server")

    print(f"Starting execution ({args.backend} backend)...")
    print("Open your browser at: http://localhost:8000")
    print("Press Ctrl+C to stop.\n")

    p.run(backend=args.backend)

if __name__ == "__main__":
    main()
