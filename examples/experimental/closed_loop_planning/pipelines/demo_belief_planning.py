# Run: pixi run -e llm demo-vlm-belief

import ast
import json
import re
import time
import cv2
import numpy as np
import rerun as rr
import threading
import queue
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from vlm_utils import BeliefVLMPlanner

from PIL import Image as PILImage

from retriever import Pipeline, Flow, Rate
from retriever.flow import Latest, io
from retriever.flow.config import EdgeConfig

# ==============================================================================
# 1. IO Types
# ==============================================================================

@io
class ImageMsg:
    frame: Optional[np.ndarray] = None
    timestamp: float = 0.0

@io
class TextMsg:
    text: str

@io
class VLACommand:
    instruction: Optional[str] = None
    mode: Optional[str] = "general"
    timestamp: float = 0.0

@io
class ReasoningInput:
    # Webcam fields
    frame: Optional[np.ndarray] = None
    timestamp: float = 0.0
    
    # Command fields
    instruction: Optional[str] = None
    mode: Optional[str] = None
    cmd_timestamp: float = 0.0


def _normalize_detections(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _quote_bare_keys(text: str) -> str:
    # Add quotes around bare keys like {box_2d: ...} -> {"box_2d": ...}
    return re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', text)


def _coerce_json(text: str) -> list:
    candidates = [text, _quote_bare_keys(text), text.replace("'", '"'), _quote_bare_keys(text.replace("'", '"'))]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return _normalize_detections(parsed)
        except json.JSONDecodeError:
            continue
    return []


def _parse_detections(raw_value) -> list:
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, dict)):
        return _normalize_detections(raw_value)
    if not isinstance(raw_value, str):
        return []

    text = raw_value.strip()
    if not text:
        return []

    detections = _coerce_json(text)
    if detections:
        return detections

    # Fallback: accept Python-style literals (single quotes, etc.)
    normalized = re.sub(r"\bnull\b", "None", text, flags=re.IGNORECASE)
    normalized = re.sub(r"\btrue\b", "True", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
    try:
        parsed = ast.literal_eval(normalized)
        return _normalize_detections(parsed)
    except (ValueError, SyntaxError):
        return []

# ==============================================================================
# 2. Flows
# ==============================================================================

class WebcamFlow(Flow[None, ImageMsg]):
    def __init__(self, device_index=0):
        self.cap = None
        self.device_index = device_index

    def init_config(self) -> dict:
        return {"device_index": self.device_index}
   
    def init(self):
        # Connect subprocess to spawned Rerun viewer
        rr.init("demo_vlm_belief", spawn=False)
        rr.connect_grpc()  # Connect to gRPC server on default port
        print(f"Opening Webcam {self.device_index}...")
        self.cap = cv2.VideoCapture(self.device_index)
       
    def step(self, _):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rr.log("webcam", rr.Image(frame_rgb))
                return ImageMsg(frame=frame_rgb, timestamp=time.time())
        return None

    def finalize(self):
        if self.cap:
            self.cap.release()
        rr.disconnect()  # Gracefully close Rerun connection

# Web Server Logic
try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except ImportError:
    class FastAPI: pass
    class BaseModel: pass
    uvicorn = None

app = FastAPI()
_cmd_queue = queue.Queue()

class CommandReq(BaseModel):
    command: str
    mode: str = "general"

@app.post("/command")
async def receive_command(req: CommandReq):
    _cmd_queue.put(req)
    return {"status": "ok", "command": req.command, "mode": req.mode}

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <body style="font-family: sans-serif; padding: 20px; background: #f4f4f9;">
            <div style="max-width: 600px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h1 style="color: #333;">Retriever Belief Space Planning</h1>
               
                <div style="margin-bottom: 20px;">
                    <label style="display: block; margin-bottom: 5px; font-weight: bold;">Mode:</label>
                    <select id="mode" style="width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #ddd;">
                        <option value="general">General Instruction</option>
                        <option value="seasoning">Seasoning Routine</option>
                        <option value="chess">Chess / Gomoku</option>
                    </select>
                </div>

                <div style="margin-bottom: 20px;">
                    <label style="display: block; margin-bottom: 5px; font-weight: bold;">Instruction:</label>
                    <input type="text" id="cmd" placeholder="e.g. 'Find the salt' or 'Next move'" style="width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #ddd; box-sizing: border-box;">
                </div>

                <button onclick="send()" style="width: 100%; padding: 12px; background: #4a90e2; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold;">
                    Send to Robot
                </button>
               
                <div id="status" style="margin-top: 20px; padding: 15px; border-radius: 6px; background: #eef2f7; color: #555;">
                    Status: Ready
                </div>
            </div>
           
            <script>
                function send() {
                    cmd = document.getElementById('cmd').value;
                    mode = document.getElementById('mode').value;
                    document.getElementById('status').innerText = "Sending [" + mode + "]: " + cmd;
                    fetch('/command', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({command: cmd, mode: mode})
                    }).then(res => {
                        document.getElementById('status').innerText = "Sent [" + mode + "]: " + cmd;
                    });
                }
            </script>
        </body>
    </html>
    """

class WebInstructionFlow(Flow[None, VLACommand]):
    def init(self):
        if uvicorn:
            self.thread = threading.Thread(
                target=uvicorn.run,
                args=(app,),
                kwargs={"host": "0.0.0.0", "port": 8000, "log_level": "info", "log_config": None},
                daemon=True
            )
            self.thread.start()
            print("[Web] Server started at http://localhost:8000")
        else:
            print("[Web] ERROR: Uvicorn not installed.")

    def step(self, _):
        try:
            req = _cmd_queue.get(timeout=0.1)
            return VLACommand(instruction=req.command, mode=req.mode, timestamp=time.time())
        except queue.Empty:
            return None

class VLMReasoningFlow(Flow[ReasoningInput, TextMsg]):
    def __init__(self, initial_task="Describe the scene", model_name="gemini-robotics-er-1.5-preview"):
        self.current_task = initial_task
        self.current_mode = "general"
        self._model_name = model_name  # Store model_name for lazy init
        self.planner = None  # Lazy init in init()
        self.last_cmd_time = 0.0

    def init_config(self):
        """Return init args for Flow reconstruction in subprocesses."""
        return {"model_name": self._model_name, "initial_task": self.current_task}

    def init(self):
        # Create BeliefVLMPlanner in subprocess to avoid pickling issues
        self.planner = BeliefVLMPlanner(model_name=self._model_name)
        print(f"[VLMReasoningFlow] Using model: {self._model_name} (with Belief)")
        # Connect subprocess to spawned Rerun viewer
        rr.init("demo_vlm_belief", spawn=False)
        rr.connect_grpc()  # Connect to gRPC server on default port
       
    def step(self, flow_input: ReasoningInput) -> Optional[TextMsg]:
        # Check if we have a new valid frame
        if flow_input.frame is None:
            return None

        # 1. Update Task/Mode if new command
        if flow_input.instruction and flow_input.cmd_timestamp > self.last_cmd_time:
            self.current_task = flow_input.instruction
            self.current_mode = flow_input.mode or "general"
            self.last_cmd_time = flow_input.cmd_timestamp
            print(f"[VLM] New Task: {self.current_task} (Mode: {self.current_mode})")
            rr.log("reasoning/task", rr.TextDocument(self.current_task))
            rr.log("reasoning/mode", rr.TextDocument(self.current_mode))

        # 2. Call the planner with timing
        start_time = time.time()
        image = PILImage.fromarray(flow_input.frame)
        print(f"[VLM] Sending request to {self.planner.model_name}...")
        result = self.planner.plan(image, self.current_task, mode=self.current_mode)
        latency_ms = (time.time() - start_time) * 1000
        
        # Log latency metrics.
        rr.log("metrics/vlm_latency_ms", rr.Scalars([latency_ms]))
        print(f"[VLM] Model: {self.planner.model_name}")
        print(f"[VLM] Latency: {latency_ms:.0f}ms | Status: {result['status']}")
       
        if result["status"] == "success":
            reasoning = result["reasoning"]
            coords = result["coordinates"]
           
            # Combine everything into one markdown visualization
            markdown_output = f"""
### ⏱️ Latency: {latency_ms:.0f} ms

###  VLM Output
{reasoning}

### 📦 Detections
```json
{coords}
```
"""
            # Log Belief State (New Feature)
            belief_mk = f"### 🧠 Current Belief\n{self.planner.current_belief}"
            rr.log("reasoning/belief", rr.TextDocument(belief_mk, media_type=rr.MediaType.MARKDOWN))

            rr.log("reasoning/output", rr.TextDocument(markdown_output, media_type=rr.MediaType.MARKDOWN))
           
            # 3. Visualize detection boxes on the image
            # We log to a separate 'inference_view' so boxes are overlaid on the EXACT image used for inference.
            try:
                # Log the snapshot image first
                rr.log("inference_view", rr.Image(flow_input.frame))

                detections = _parse_detections(coords)
                if detections:
                    print(f"[VLM] Detections: {len(detections)} boxes")
                elif coords:
                    print("[VLM] Detections parse failed; falling back to raw text.")
                
                # Debugging: show raw text if reasoning is empty
                if not reasoning.strip():
                     print(f"[VLM] Raw Output: {result['raw_text']}")
                else:
                     print(f"[VLM] Reasoning: {reasoning}\n")
                
                if detections and isinstance(detections, list):
                    h, w = flow_input.frame.shape[:2]
                    boxes = []
                    labels = []
                    for det in detections:
                        if not isinstance(det, dict):
                            continue
                        if "box_2d" in det:
                            # box_2d is [ymin, xmin, ymax, xmax] in 0-1000 normalized
                            ymin, xmin, ymax, xmax = det["box_2d"]
                            # Convert to pixel coords
                            x1 = int(xmin * w / 1000)
                            y1 = int(ymin * h / 1000)
                            x2 = int(xmax * w / 1000)
                            y2 = int(ymax * h / 1000)
                            boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h]
                            labels.append(det.get("label", "detection"))
                    
                    if boxes:
                        rr.log("inference_view", rr.Boxes2D(
                            array=boxes,
                            array_format=rr.Box2DFormat.XYWH,
                            labels=labels,
                            colors=[(0, 255, 0)]  # Green boxes
                        ))
            except (KeyError, TypeError, ValueError) as e:
                print(f"[VLM] Vis Error: {e}")
           
            print(f"[VLM] Reasoning: {reasoning[:80]}...")
           
            return TextMsg(text=coords)
        elif result["status"] == "rate_limited":
            print(f"[VLMReasoning] {result['reasoning']}")
        elif result["status"] == "busy":
            pass
        else:
            print(f"[VLMReasoning] Error: {result.get('message')}")
           
        return None


    def finalize(self):
        rr.disconnect()  # Gracefully close Rerun connection

# ==============================================================================
# 3. Pipeline
# ==============================================================================
if __name__ == "__main__":
    import argparse
    import retriever

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model", type=str, default="gemini-robotics-er-1.5-preview", help="API model ID")
    args = parser.parse_args()

    # Initialize Rerun viewer (main process spawns, subprocesses connect)
    rr.init("demo_vlm_belief", spawn=True)
    retriever.init("demo_belief_planning")

    with Pipeline("demo_vlm_belief") as pipe:
        # High-rate webcam and visual logging
        cam = WebcamFlow(args.device) @ Rate(10)
        web = WebInstructionFlow() @ Rate(5)

        # Low-rate VLM analysis (with larger queue to avoid drop warnings)
        planner = VLMReasoningFlow(model_name=args.model) @ Rate(0.2, on_lag="drop")

        # Connect using .then() and explicit field mapping
        cam.then(planner, map={"frame": "frame", "timestamp": "timestamp"}, sync=Latest(), edge_config={
            "frame": EdgeConfig(qsize=100, on_full="drop"),
            "timestamp": EdgeConfig(qsize=100)
        })
        web.then(planner, map={"instruction": "instruction", "mode": "mode", "timestamp": "cmd_timestamp"}, sync=Latest(), edge_config={
            "instruction": EdgeConfig(qsize=100),
            "mode": EdgeConfig(qsize=100),
            "cmd_timestamp": EdgeConfig(qsize=100)
        })

    try:
        pipe.visualize(open_browser=True)
        pipe.run(backend="dora", duration=30)
    except KeyboardInterrupt:
        pass
