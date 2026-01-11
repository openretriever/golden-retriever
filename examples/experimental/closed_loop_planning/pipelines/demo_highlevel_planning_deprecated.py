# Deprecated: legacy high-level planning pipeline.
# Run: pixi run -e llm demo-highlevel-planning-deprecated
#
# ============================================================================
# High-Level Planning Pipeline with VLM Language Instructions
# ============================================================================
#
# This demo implements a high-level planning pipeline where:
# 1. VLM generates a language plan (list of instructions)
# 2. Plan is displayed in Rerun (one instruction per line)
# 3. Web interface allows instruction injection
# 4. Execution Monitor tracks state (IDLE/PLANNING/EXECUTING)
#
# Architecture:
#
#   WebcamFlow (10Hz)
#        │
#        ▼
#   HighLevelPlannerFlow @ Rate(0.2Hz)  ◄──── WebInstructionFlow
#        │
#        │ plan: List[str]
#        ▼
#   ExecutionMonitorFlow @ Rate(1Hz)
#        │
#        ▼
#   Rerun Visualization (TextDocument)
#
# ============================================================================

import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import rerun as rr

from retriever.flow import Flow, Latest, Pipeline, Rate, io

# Import the BeliefVLMPlanner from vlm_utils
import sys
import os

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.experimental.closed_loop_planning.pipelines.vlm_utils import BeliefVLMPlanner


# ============================================================================
# Data Types
# ============================================================================

class ExecutionState(Enum):
    """State machine for execution monitor."""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"


@io
@dataclass
class PlannerInput:
    """Input to the high-level planner."""
    frame: Optional[bytes] = None
    task: str = ""
    timestamp: float = 0.0


@io
@dataclass
class PlanOutput:
    """Output from the high-level planner."""
    plan: List[str] = field(default_factory=list)  # List of language instructions
    reasoning: str = ""
    belief_update: str = ""
    status: str = "idle"


@io
@dataclass
class MonitorOutput:
    """Output from execution monitor."""
    state: ExecutionState = ExecutionState.IDLE
    current_step: int = 0
    current_instruction: str = ""
    plan_display: str = ""  # Formatted for Rerun


# ============================================================================
# Web Instruction Flow (FastAPI)
# ============================================================================

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except ImportError:
    pass

app = FastAPI()

class InstructionRequest(BaseModel):
    instruction: str

_instruction_queue = queue.Queue()

@app.post("/instruction")
async def receive_instruction(req: InstructionRequest):
    _instruction_queue.put((req.instruction, time.time()))
    return {"status": "ok", "instruction": req.instruction}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>High-Level Planning</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            input { width: 100%; padding: 12px; font-size: 16px; margin: 10px 0; }
            button { padding: 12px 24px; font-size: 16px; background: #4CAF50; color: white; border: none; cursor: pointer; }
            button:hover { background: #45a049; }
            #status { margin-top: 20px; padding: 10px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <h1>🤖 High-Level Planning Interface</h1>
        <p>Enter a task instruction for the VLM planner:</p>
        <input type="text" id="instruction" placeholder="e.g., Clean the table" />
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


class WebInstructionFlow(Flow[None, PlannerInput]):
    """Web interface for instruction injection."""
    
    def __init__(self, name: str = "WebInstructionFlow"):
        self.name = name
    
    def init(self):
        self.server_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={
                "host": "0.0.0.0",
                "port": 8000,
                "log_level": "warning",
                "use_colors": False,
                "log_config": None
            },
            daemon=True
        )
        self.server_thread.start()
        print("[Web] Server started at http://localhost:8000")
    
    def step(self, _) -> Optional[PlannerInput]:
        try:
            instruction, timestamp = _instruction_queue.get(timeout=0.1)
            return PlannerInput(task=instruction, timestamp=timestamp)
        except queue.Empty:
            return None


# ============================================================================
# High-Level Planner Flow
# ============================================================================

class HighLevelPlannerFlow(Flow[PlannerInput, PlanOutput]):
    """
    VLM-based high-level planner that generates a list of language instructions.
    """
    
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        # Allow env var override since dora might re-instantiate with default args
        self.model_name = os.getenv("VLM_MODEL", model)
        self.planner: Optional[BeliefVLMPlanner] = None
        self.current_task = "Observe the scene and describe what you see."
    
    def init(self):
        # Initialize VLM planner with special prompt for language plan output
        self.planner = BeliefVLMPlanner(model_name=self.model_name)
        
        # Override system prompt to request structured plan output
        self.planner.system_prompt = """You are a robot task planner. Given an image and a task, output:

1. REASONING: Brief analysis of the current scene.
2. PLAN: A numbered list of high-level language instructions to accomplish the task.
   Each instruction should be a single, specific action that a robot could understand.
3. BELIEF_UPDATE: What changed in your understanding of the world.

Format your response as JSON:
{
    "reasoning": "I see a table with a cup and a plate...",
    "plan": [
        "1. Locate the cup on the table",
        "2. Grasp the cup",
        "3. Move cup to the sink",
        "4. Release the cup"
    ],
    "belief_update": "Cup is now in the sink."
}
"""
        rr.init("highlevel_planning", spawn=False)
        rr.connect_grpc()
        print(f"[Planner] Using model: {self.model_name}")
    
    def step(self, inp: PlannerInput) -> PlanOutput:
        if inp is None or (inp.frame is None and not inp.task):
            return PlanOutput(status="waiting")
        
        # Update task if provided
        if inp.task:
            self.current_task = inp.task
            print(f"[Planner] New task: {self.current_task}")
        
        # Skip if no frame
        if inp.frame is None:
            return PlanOutput(status="no_frame")
        
        # Generate plan using VLM
        start = time.time()
        try:
            from PIL import Image
            import io as iolib
            image = Image.open(iolib.BytesIO(inp.frame))
            result = self.planner.plan(image, self.current_task)
            latency_ms = (time.time() - start) * 1000
            
            # Defensive: ensure result is a dict
            if not isinstance(result, dict):
                print(f"[Planner] Unexpected result type: {type(result)}")
                return PlanOutput(status="error")
            
            # Parse response
            if result.get("status") == "success":
                raw_text = result.get("raw_text", "")
                parsed = self._parse_plan(raw_text)
                
                # Log to Rerun
                rr.log("metrics/planner_latency_ms", rr.Scalars([latency_ms]))
                
                return PlanOutput(
                    plan=parsed.get("plan", []),
                    reasoning=parsed.get("reasoning", ""),
                    belief_update=parsed.get("belief_update", ""),
                    status="success"
                )
            else:
                return PlanOutput(status=result.get("status", "error"))
                
        except Exception as e:
            print(f"[Planner] Error: {e}")
            return PlanOutput(status="error")
    
    def _parse_plan(self, raw_text: str) -> dict:
        """Parse VLM response into structured plan."""
        # Try JSON parsing first
        try:
            # Find JSON block
            if "```json" in raw_text:
                json_str = raw_text.split("```json")[1].split("```")[0]
            elif "[" in raw_text and ("{" not in raw_text or raw_text.index("[") < raw_text.index("{")):
                # JSON array
                start = raw_text.index("[")
                end = raw_text.rindex("]") + 1
                json_str = raw_text[start:end]
            elif "{" in raw_text:
                start = raw_text.index("{")
                end = raw_text.rindex("}") + 1
                json_str = raw_text[start:end]
            else:
                return {"reasoning": raw_text, "plan": [], "belief_update": ""}
            
            parsed = json.loads(json_str)
            
            # Handle list (VLM output is just a plan array)
            if isinstance(parsed, list):
                return {"reasoning": "", "plan": parsed, "belief_update": ""}
            return parsed
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract numbered lines as plan
            lines = raw_text.split("\n")
            plan = [line.strip() for line in lines if line.strip() and line.strip()[0].isdigit()]
            return {"reasoning": raw_text[:200], "plan": plan, "belief_update": ""}


# ============================================================================
# Execution Monitor Flow (VLM-based task completion checking)
# ============================================================================

@io
@dataclass
class MonitorInput:
    """Input to execution monitor - needs frame + plan."""
    frame: Optional[bytes] = None
    plan: List[str] = field(default_factory=list)
    reasoning: str = ""
    belief_update: str = ""
    status: str = ""


class VLMExecutionMonitorFlow(Flow[MonitorInput, MonitorOutput]):
    """
    VLM-based execution monitor that checks if current instruction is complete.
    
    Uses VLM to analyze webcam feed and determine if the current step
    in the plan has been accomplished visually.
    """
    
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        # Allow env var override since dora might re-instantiate with default args
        self.model_name = os.getenv("VLM_MODEL", model)
        self.vlm: Optional[BeliefVLMPlanner] = None
    
    def init(self):
        self.state = ExecutionState.IDLE
        self.current_plan: List[str] = []
        self.current_step = 0
        
        # Initialize VLM for task completion checking
        self.vlm = BeliefVLMPlanner(model_name=self.model_name)
        self.vlm.system_prompt = """You are a visual task completion checker.
Given an image and a task instruction, determine if the task appears to be complete.

Respond with ONLY one of:
- "COMPLETE" if the task is visibly finished
- "IN_PROGRESS" if the task is being executed but not done
- "NOT_STARTED" if there's no evidence the task has begun

Be conservative - only say COMPLETE if you're confident."""
        
        rr.init("highlevel_planning", spawn=False)
        rr.connect_grpc()
        print(f"[Monitor] VLM-based execution monitoring with {self.model_name}")
    
    def step(self, inp: MonitorInput) -> MonitorOutput:
        # Update plan if received
        if inp.plan and inp.plan != self.current_plan:
            self.current_plan = inp.plan
            self.current_step = 0
            self.state = ExecutionState.EXECUTING
            print(f"[Monitor] New plan with {len(inp.plan)} steps")
        
        # If we have a plan and frame, check if current step is complete
        if (self.current_plan and 
            self.current_step < len(self.current_plan) and 
            inp.frame is not None and
            self.state == ExecutionState.EXECUTING):
            
            current_instruction = self.current_plan[self.current_step]
            
            # Check completion using VLM
            try:
                from PIL import Image
                import io as iolib
                image = Image.open(iolib.BytesIO(inp.frame))
                
                check_prompt = f"Current instruction: {current_instruction}"
                result = self.vlm.plan(image, check_prompt)
                
                if result.get("status") == "success":
                    response = result.get("raw_text", "").upper()
                    
                    if "COMPLETE" in response:
                        print(f"[Monitor] ✓ Step {self.current_step + 1} COMPLETE: {current_instruction}")
                        self.current_step += 1
                        
                        # Check if all steps done
                        if self.current_step >= len(self.current_plan):
                            self.state = ExecutionState.IDLE
                            print("[Monitor] ✓✓ All steps complete!")
                    elif "IN_PROGRESS" in response:
                        print(f"[Monitor] ... Step {self.current_step + 1} in progress")
                        
            except Exception as e:
                print(f"[Monitor] VLM check error: {e}")
        
        # Format plan for display
        plan_lines = []
        for i, step in enumerate(self.current_plan):
            if i < self.current_step:
                marker = "✓"  # Completed
            elif i == self.current_step:
                marker = "▶"  # Current
            else:
                marker = "○"  # Pending
            plan_lines.append(f"{marker} {step}")
        
        plan_display = "\n".join(plan_lines) if plan_lines else "(No plan)"
        
        # Log to Rerun
        rr.log("planning/state", rr.TextLog(self.state.value))
        rr.log("planning/plan", rr.TextDocument(plan_display))
        rr.log("planning/current_step", rr.Scalars([float(self.current_step)]))
        if inp.reasoning:
            rr.log("planning/reasoning", rr.TextDocument(inp.reasoning[:500]))
        if inp.belief_update:
            rr.log("planning/belief", rr.TextDocument(inp.belief_update))
        
        current_instruction = self.current_plan[self.current_step] if self.current_step < len(self.current_plan) else ""
        
        return MonitorOutput(
            state=self.state,
            current_step=self.current_step,
            current_instruction=current_instruction,
            plan_display=plan_display
        )


# ============================================================================
# Webcam Flow (reuse from existing)
# ============================================================================

@io
@dataclass
class FrameOutput:
    frame: Optional[bytes] = None
    timestamp: float = 0.0


class WebcamFlow(Flow[None, FrameOutput]):
    """Captures frames from webcam."""
    
    def __init__(self, device: int = 0, name: str = "WebcamFlow"):
        self.name = name
        self.device = device
        self.cap = None
    
    def init(self):
        import cv2
        self.cap = cv2.VideoCapture(self.device)
        print(f"[Webcam] Opened device {self.device}")
    
    def step(self, _) -> FrameOutput:
        import cv2
        if self.cap is None:
            return FrameOutput()
        
        ret, frame = self.cap.read()
        if not ret:
            return FrameOutput()
        
        # Encode as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        return FrameOutput(frame=buffer.tobytes(), timestamp=time.time())


# ============================================================================
# Pipeline Construction
# ============================================================================

def build_pipeline(model: str) -> Pipeline:
    pipe = Pipeline("highlevel_planning")
    
    with pipe:
        # Sources
        webcam = WebcamFlow() @ Rate(10)
        web_input = WebInstructionFlow() @ Rate(10)
        
        # Planner (slow, VLM inference)
        planner = HighLevelPlannerFlow(model=model) @ Rate(0.2)
        
        # Monitor (VLM-based task completion checking)
        monitor = VLMExecutionMonitorFlow(model=model) @ Rate(1)
        
        # Connect webcam -> planner
        pipe.connect(webcam, planner, map={"frame": "frame", "timestamp": "timestamp"}, sync=Latest())
        
        # Connect web input -> planner (for task updates)
        pipe.connect(web_input, planner, map={"task": "task"}, sync=Latest())
        
        # Connect planner -> monitor (plan + reasoning)
        pipe.connect(planner, monitor, map={"plan": "plan", "reasoning": "reasoning", "belief_update": "belief_update", "status": "status"}, sync=Latest())
        
        # Connect webcam -> monitor (for VLM completion checking)
        pipe.connect(webcam, monitor, map={"frame": "frame"}, sync=Latest())
    
    return pipe


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="High-Level Planning Pipeline")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="VLM model to use")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    args = parser.parse_args()
    
    # Set env var for subprocesses
    os.environ["VLM_MODEL"] = args.model
    
    pipe = build_pipeline(args.model)
    
    print("=" * 60)
    print("High-Level Planning Pipeline")
    print("=" * 60)
    print(f"Model: {args.model}")
    print("Open browser: http://localhost:8000")
    print("View Rerun: rerun --port 9876")
    print("=" * 60)
    
    pipe.visualize()
    pipe.run(backend="dora", duration=args.duration)


if __name__ == "__main__":
    main()
