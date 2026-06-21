# Run: pixi run -e llm demo-highlevel-planning
#
# ============================================================================
# Modular High-Level Planning Pipeline
# ============================================================================
#
# This pipeline mirrors the architecture of demo_manipulation.py but adapted
# for VLM-based high-level planning.
#
# Architecture:
#
#   WebcamFlow (10Hz) --> PerceptionFlow --> BeliefUpdaterFlow
#                                                   │
#   WebInstructionFlow ----------------------------►│
#                                                   │
#   VLMTaskPlannerFlow <--(state)-------------------┤
#        │                                          │
#        ▼                                          │
#   VLMExecutionMonitorFlow <--(plan)---------------┤
#        │
#        └-------(status summary only)
#
# ============================================================================

import argparse
import queue
import socket
import threading
import time
import os
import sys

# Ensure project root in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import retriever
from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, io
from dataclasses import dataclass
from typing import Optional

# Reuse Standard Flows - VLM-compatible versions
from examples.experimental.closed_loop_planning.flows.perception_vlm import (
    VLMPerceptionFlow,
)
from examples.experimental.closed_loop_planning.flows.belief_updater_vlm import (
    VLMBeliefUpdaterFlow,
)

# New VLM Flows
from examples.experimental.closed_loop_planning.flows.planner_vlm import (
    VLMTaskPlannerFlow,
)
from examples.experimental.closed_loop_planning.flows.monitor_vlm import (
    VLMExecutionMonitorFlow,
)

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except ImportError:
    uvicorn = None
    FastAPI = None
    HTMLResponse = None
    BaseModel = object


@io
@dataclass
class CameraOutput:
    data: dict
    frame: Optional[bytes] = None


@io
@dataclass
class WebInstructionOutput:
    task: str
    timestamp: float


_instruction_queue = queue.Queue()
_web_app = FastAPI() if FastAPI else None

if _web_app:

    class InstructionRequest(BaseModel):
        instruction: str

    @_web_app.post("/instruction")
    async def receive_instruction(req: InstructionRequest):
        _instruction_queue.put((req.instruction, time.time()))
        return {"status": "ok", "instruction": req.instruction}

    @_web_app.get("/", response_class=HTMLResponse)
    async def get_index():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>High-Level Planning</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 640px; margin: 40px auto; padding: 20px; }
                h1 { color: #333; }
                input { width: 100%; padding: 12px; font-size: 16px; margin: 10px 0; }
                button { padding: 12px 24px; font-size: 16px; background: #2f6fed; color: white; border: none; cursor: pointer; }
                button:hover { background: #2459c4; }
                #status { margin-top: 20px; padding: 10px; background: #f0f0f0; }
            </style>
        </head>
        <body>
            <h1>High-Level Planning Interface</h1>
            <p>Enter a task instruction for the VLM planner:</p>
            <input type="text" id="instruction" placeholder="e.g., Clear the table" />
            <button onclick="sendInstruction()">Send</button>
            <div id="status"></div>
            <script>
                async function sendInstruction() {
                    const instruction = document.getElementById('instruction').value;
                    const response = await fetch('/instruction', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({instruction: instruction})
                    });
                    const data = await response.json();
                    document.getElementById('status').innerHTML =
                        '<strong>Sent:</strong> ' + data.instruction;
                }
            </script>
        </body>
        </html>
        """


class WebInstructionFlow(Flow[None, WebInstructionOutput]):
    """Web interface for instruction injection."""

    def __init__(self, name: str = "WebInstructionFlow"):
        self.name = name

    def init(self):
        if uvicorn is None or _web_app is None:
            print("[Web] ERROR: Uvicorn/FastAPI not installed.")
            return

        self.server_thread = threading.Thread(
            target=uvicorn.run,
            args=(_web_app,),
            kwargs={
                "host": "0.0.0.0",
                "port": 8000,
                "log_level": "warning",
                "use_colors": False,
                "log_config": None,
            },
            daemon=True,
        )
        self.server_thread.start()
        print("[Web] Server started at http://localhost:8000")

    def step(self, _) -> Optional[WebInstructionOutput]:
        try:
            instruction, timestamp = _instruction_queue.get(timeout=0.1)
            return WebInstructionOutput(task=instruction, timestamp=timestamp)
        except queue.Empty:
            return None


class WebcamSourceFlow(Flow[None, CameraOutput]):
    """Captures frames from webcam and outputs dict for PerceptionFlow."""

    def __init__(self, device: int = 0, name: str = "WebcamSource"):
        self.name = name
        self.device = device
        self.cap = None
        self._warned_black_frame = False

    def init_config(self) -> dict:
        return {"device": self.device, "name": self.name}

    def init(self):
        import cv2

        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            print(f"[{self.name}] Failed to open device {self.device}")
        else:
            print(f"[{self.name}] Opened device {self.device}")

    def step(self, _) -> CameraOutput:
        import cv2

        if self.cap is None:
            return CameraOutput(data={}, frame=None)

        ret, frame = self.cap.read()
        if not ret:
            return CameraOutput(data={}, frame=None)

        if not self._warned_black_frame and frame is not None and frame.max() == 0:
            print(
                f"[{self.name}] Frame appears black; check camera permissions or device index."
            )
            self._warned_black_frame = True

        # Encode as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        return CameraOutput(
            data={"rgb": frame_bytes, "timestamp": time.time()}, frame=frame_bytes
        )


def build_modular_pipeline(model: str, initial_task: str, device: int) -> Pipeline:
    pipe = Pipeline("highlevel_planning")

    with pipe:
        # --- 1. Sources ---
        # Webcam captures at 10Hz
        webcam = WebcamSourceFlow(name="Webcam", device=device) @ Rate(10)

        # Web UI for instructions
        web_input = WebInstructionFlow(name="WebInput") @ Rate(10)

        # --- 2. Core Modules ---

        # Perception: Wraps raw data (Triggered by webcam)
        perception = VLMPerceptionFlow(name="Perception") @ Trigger("data")

        # Belief: Updates state (Triggered by perception)
        belief = VLMBeliefUpdaterFlow(name="BeliefUpdater") @ Trigger("observation")

        # Planner: Generates plan (Triggered by belief update)
        # Note: We rate limit this inside the flow, or we could use Rate() here
        # but Trigger(\"state\") ensures we plan on fresh data.
        planner = VLMTaskPlannerFlow(
            name="VLMPlanner", model=model, initial_task=initial_task
        ) @ Trigger("state")

        # Monitor: Checks execution (Triggered by webcam frame for visual check)
        # We trigger on 'frame' so we check frequently.
        monitor = VLMExecutionMonitorFlow(name="VLMMonitor", model=model) @ Trigger(
            "frame",
            "plan",
            "reasoning",
            "belief_update",
            "task",
            "timestamp",
        )

        # --- 3. Connections ---

        # Webcam -> Perception (Raw Data)
        # WebcamSourceFlow outputs "data" (dict) and "frame" (bytes)
        # VLMPerceptionFlow expects "data" (dict)
        pipe.connect(webcam, perception, map={"data": "data"}, sync=Latest())

        # Perception -> Belief (Observation + Raw Observation)
        # VLMBeliefUpdater expects "observation", "raw_observation"
        # VLMPerceptionFlow outputs "state", "raw_observation"
        pipe.connect(
            perception,
            belief,
            map={"state": "observation", "raw_observation": "raw_observation"},
            sync=Latest(),
        )

        # WebInput -> Planner (Task/Instruction)
        # Map 'task' from WebInstructionFlow to 'task' in PlannerInput
        pipe.connect(
            web_input,
            planner,
            map={"task": "task", "timestamp": "timestamp"},
            sync=Latest(),
        )

        # Webcam -> Monitor (Frame for visual check)
        pipe.connect(webcam, monitor, map={"frame": "frame"}, sync=Latest())

        # Webcam -> Planner (Frame passthrough so planner doesn't rely on belief serialization)
        pipe.connect(webcam, planner, map={"frame": "frame"}, sync=Latest())

        # Belief -> Planner (State)
        pipe.connect(belief, planner, map={"belief": "state"}, sync=Latest())

        # Belief -> Monitor (State, mainly for visual context if needed)
        pipe.connect(belief, monitor, map={"belief": "state"}, sync=Latest())

        # Planner -> Monitor (The Plan)
        pipe.connect(
            planner,
            monitor,
            map={
                "plan": "plan",
                "task": "task",
                "reasoning": "reasoning",
                "belief_update": "belief_update",
            },
            sync=Latest(),
        )

        # WebInput -> Monitor (Task context for status display)
        pipe.connect(
            web_input,
            monitor,
            map={"task": "task", "timestamp": "timestamp"},
            sync=Latest(),
        )

    return pipe


def _is_rerun_viewer_running(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description="Modular High-Level Planning Pipeline")
    parser.add_argument(
        "--model", default="gemini-2.5-flash-lite", help="VLM model to use"
    )
    parser.add_argument(
        "--task",
        default="",
        help="Initial task for the planner (can also be sent via the web UI)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Webcam device index",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0, help="Duration in seconds"
    )
    parser.add_argument(
        "--rerun-port",
        type=int,
        default=9876,
        help="Rerun gRPC port (use a dedicated port if other demos are running)",
    )
    rerun_spawn_group = parser.add_mutually_exclusive_group()
    rerun_spawn_group.add_argument(
        "--rerun-spawn",
        action="store_true",
        help="Force spawn a Rerun viewer",
    )
    rerun_spawn_group.add_argument(
        "--rerun-no-spawn",
        action="store_true",
        help="Never spawn a Rerun viewer",
    )
    parser.add_argument(
        "--no-open-browser",
        dest="open_browser",
        action="store_false",
        help="Do not open the pipeline graph in a browser",
    )
    parser.set_defaults(open_browser=True)
    args = parser.parse_args()

    # Initialize global retriever (backend config)
    rerun_host = "127.0.0.1"
    rerun_port = args.rerun_port
    rerun_port_explicit = any(arg.startswith("--rerun-port") for arg in sys.argv)
    rerun_running = _is_rerun_viewer_running(rerun_host, rerun_port)

    if args.rerun_no_spawn:
        spawn_viewer = False
    elif args.rerun_spawn:
        spawn_viewer = True
        if rerun_running:
            new_port = _find_free_port(rerun_host)
            print(
                f"[Rerun] Port {rerun_port} is in use; spawning on {new_port} instead."
            )
            rerun_port = new_port
            rerun_running = False
    else:
        # Default: connect to existing viewer if running, otherwise spawn
        spawn_viewer = not rerun_running
        if rerun_running:
            print(f"[Rerun] Connecting to existing viewer at {rerun_host}:{rerun_port}")

    retriever.init(
        backend="dora",
        backend_config={
            "dora_timeout": 30,
            "rerun_config": {
                "connect_addr": f"{rerun_host}:{rerun_port}",
                "spawn": spawn_viewer,
            },
        },
    )

    pipe = build_modular_pipeline(args.model, args.task, args.device)

    print("=" * 60)
    print("Modular High-Level Planning Pipeline")
    print("=" * 60)
    print(f"Model: {args.model}")
    print("Open browser: http://localhost:8000")
    print(f"View Rerun: pixi run -e llm rerun --port {rerun_port}")
    if not args.task:
        print(
            "Planner task: (none) — use the web UI or pass --task to trigger VLM planning"
        )
    print("=" * 60)

    pipe.visualize(open_browser=args.open_browser)
    pipe.run(duration=args.duration)


if __name__ == "__main__":
    main()
