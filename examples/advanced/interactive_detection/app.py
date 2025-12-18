"""
Retriever Advanced Example: Interactive Vision Detection

This demo performs real-time open-vocabulary object detection using OWL-ViT.
Users can input natural language prompts (e.g., "detect the coffee mug") via 
a web interface to update the detection targets dynamcially.

Architecture:
    Camera --(image)--> Detector --(result)--> WebNode
                          ^                      |
                          |__(prompt)____________|

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
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
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
    Pipeline, Rate, Flow, flow_io
)

# =============================================================================
# 1. Message Types (Flow IO)
# =============================================================================

@flow_io
@dataclass
class FrameMsg:
    image: np.ndarray

@flow_io
@dataclass
class PromptMsg:
    text: str

@flow_io
@dataclass
class DetectorInput:
    image: Optional[np.ndarray] = None
    prompt: Optional[str] = None

@flow_io
@dataclass
class DetectionResult:
    image: np.ndarray
    detections: List[Dict[str, Any]]
    latency_ms: float
    prompt: str

# =============================================================================
# 2. Models & Utilities
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
# 3. Flows
# =============================================================================

class WebcamFlow(Flow[None, FrameMsg]):
    """Source flow that captures frames from the webcam."""
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        
    def init(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"[WebcamFlow] Warning: Could not open webcam {self.camera_id}. Frame stream will be empty.")
                self.cap = None
        except Exception as e:
            print(f"[WebcamFlow] Error initializing camera: {e}")
            self.cap = None
        
    def run(self, _):
        if self.cap is None:
            time.sleep(1.0) # Prevent busy loop if camera failed
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
        return FrameMsg(image=frame)
    
    def init_config(self):
        return {"camera_id": self.camera_id}


class VisionDetectorFlow(Flow[DetectorInput, DetectionResult]):
    """Transform flow that performs object detection. Maintains state of current prompt."""
    def __init__(self, threshold=0.1, initial_prompt="person, coffee mug, cell phone"):
        self.threshold = threshold
        self.initial_prompt = initial_prompt
        self.detector = None
        self.current_prompt = None

    def init(self):
        self.detector = VisionDetector()
        self.current_prompt = self.initial_prompt

    def run(self, input: DetectorInput):
        if input is None: 
            return None

        # Update prompt state if provided
        if input.prompt is not None:
             # Basic cleaning
             p = input.prompt.strip()
             if p:
                 self.current_prompt = p
                 print(f"[Detector] Prompt updated: {self.current_prompt}")

        if input.image is None:
            return None
        
        # Guard against no prompt ever
        if not self.current_prompt:
             # Should not happen given init, but fallback
             return DetectionResult(
                 image=input.image, 
                 detections=[], 
                 latency_ms=0.0, 
                 prompt=""
             )

        start_time = time.time()
        queries = [q.strip() for q in self.current_prompt.split(",")]
        detections = self.detector.detect(input.image, queries, threshold=self.threshold)
        
        latency = (time.time() - start_time) * 1000
        return DetectionResult(
            image=input.image,
            detections=detections,
            latency_ms=latency,
            prompt=self.current_prompt
        )

    def init_config(self):
        return {"threshold": self.threshold, "initial_prompt": self.initial_prompt}


class WebInteractiveNode(Flow[DetectionResult, DetectorInput]):
    """
    Node that hosts FastAPI:
    - SINK for DetectionResult (updates video stream)
    - SOURCE for DetectorInput (emits user frames OR prompt updates)
    """
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        
        # Runtime attributes
        self.latest_frame = None
        self.latest_data = {}
        self._frame_lock = None
        self._input_queue = None
        self.app = None
        self.server_thread = None
        
    def init(self):
        self._frame_lock = threading.Lock()
        self._input_queue = queue.Queue()
        
        self.app = FastAPI(title="Retriever Vision Detection Demo")
        self.setup_routes()
        
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

    def setup_routes(self):
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        self.templates = Jinja2Templates(directory=static_dir)

        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})

        @self.app.websocket("/ws/camera")
        async def websocket_camera(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_bytes()
                    # Decode JPEG to numpy
                    nparr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is not None and self._input_queue:
                         # Prioritize image input
                         self._input_queue.put(DetectorInput(image=img))
            except WebSocketDisconnect:
                print("[WebInteractiveNode] Client camera disconnected")
            except Exception as e:
                print(f"[WebInteractiveNode] WS Error: {e}")

        @self.app.get("/video_feed")
        async def video_feed():
            return StreamingResponse(
                self.gen_frames(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

        @self.app.post("/update_prompt")
        async def update_prompt(data: Dict[str, str]):
            prompt = data.get("prompt", "")
            if self._input_queue:
                self._input_queue.put(DetectorInput(prompt=prompt))
            return {"status": "success", "new_prompt": prompt}

        @self.app.get("/stats")
        async def get_stats():
            if self._frame_lock is None: return {}
            with self._frame_lock:
                lat = self.latest_data.get("latency_ms")
                if lat is None: lat = 0.0
                
                dets = self.latest_data.get("detections")
                if dets is None: dets = []
                
                return {
                    "latency_ms": round(lat, 1),
                    "prompt": self.latest_data.get("prompt", ""),
                    "detections": len(dets)
                }

    def gen_frames(self):
        while True:
            if self._frame_lock is None:
                time.sleep(0.1)
                continue
                
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
        # Disable uvicorn's default logging config to avoid interference with multiprocessing
        uvicorn.run(self.app, host=self.host, port=self.port, log_config=None)

    def run(self, input: DetectionResult) -> DetectorInput:
        # 1. Sink: Update internal state
        if self._frame_lock:
            with self._frame_lock:
                self.latest_frame = input.image
                self.latest_data = {
                    "detections": input.detections,
                    "latency_ms": input.latency_ms,
                    "prompt": input.prompt
                }
        
        # 2. Source: Check for new inputs (frames OR prompts)
        try:
            if self._input_queue:
                return self._input_queue.get_nowait()
        except queue.Empty:
            pass
            
        return None

    def init_config(self):
        return {"host": self.host, "port": self.port}

# =============================================================================
# 4. Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Interactive Vision Detection Demo")
    parser.add_argument("--backend", type=str, default="dora", 
                        choices=["multiprocessing", "dora"], help="Runtime backend")
    parser.add_argument("--client-camera", action="store_true", 
                        help="Use client (browser) camera instead of server webcam")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("Retriever Advanced Example: Interactive Vision Detection")
    print("="*60 + "\n")

    # Build Pipeline
    p = Pipeline("vision_vla_interactive")
    
    with p:
        # Configuration
        # Reduced rate (2Hz) to ensure CPU inference (OWL-ViT) can keep up without filling queues.
        detector = VisionDetectorFlow() @ Rate(hz=2)
        
        # Web Node runs faster to keep UI responsive
        web_node = WebInteractiveNode() @ Rate(hz=30) 

        if args.client_camera:
            print(">> Mode: Client Camera (Browser -> Server -> Detection)")
            # WebNode is the Source of images (via WS)
            # WebNode outputs DetectorInput -> Detector
            p.connect(web_node, detector, qsize=10)
            
            # Detector -> WebNode (Sink)
            p.connect(detector, web_node, qsize=1)
            
        else:
            print(">> Mode: Server Webcam")
            camera = WebcamFlow() @ Rate(hz=2)
            
            # Camera -> Detector
            p.connect(camera, detector, map={"image": "image"}, qsize=1)
            
            # Detector -> WebNode (Sink)
            p.connect(detector, web_node, qsize=1)
            
            # WebNode (Prompt) -> Detector
            # Note: WebNode now outputs DetectorInput, so we don't need 'map' if fields match, 
            # but DetectorInput contains optional image/prompt. 
            # VisionDetectorFlow expects DetectorInput. Compatible!
            p.connect(web_node, detector, qsize=10)
        
    print(f"Starting execution ({args.backend} backend)...")
    print(f"Open your browser at: http://localhost:8000")
    if args.client_camera:
        print("Enable 'Use Browser Camera' in the UI to stream video.")
    print("Press Ctrl+C to stop.\n")

    p.run(backend=args.backend)

if __name__ == "__main__":
    main()
