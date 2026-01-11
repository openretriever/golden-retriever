from typing import Optional, List, Any
from retriever.flow import Flow
from ..types.flow_types import VLMPlannerInput, PlannerOutput
from ..pipelines.vlm_utils import BeliefVLMPlanner
import time
import os

class VLMTaskPlannerFlow(Flow[VLMPlannerInput, PlannerOutput]):
    """
    VLM-based task planner that generates language plans properly integrated 
    into the belief-based pipeline.
    
    Acts as a drop-in replacement/alternative to TaskPlannerFlow (A*).
    """
    
    def __init__(
        self,
        name: str = "VLMTaskPlanner",
        model: str = "gemini-2.5-flash-lite",
        initial_task: str = "",
    ):
        self.name = name
        # Allow env var override since dora might re-instantiate with default args
        self.model_name = os.getenv("VLM_MODEL", model)
        self.planner: Optional[BeliefVLMPlanner] = None
        self.current_task = initial_task
        self.last_plan_time = 0
        self.last_task_timestamp = 0.0
        self.min_plan_interval = 2.0  # Prevent VLM spamming
        self.current_plan: List[str] = []
        self.last_reasoning = ""
        self.last_belief_update = ""
        self.plan_attempt_count = 0
        self.plan_history: List[dict] = []
        self.max_plan_history = 5
        self._warned_no_task = False
        self._warned_no_state = False
        self._warned_no_frame = False
        self._warned_bad_frame = False
        
    def init_config(self) -> dict:
        return {
            "name": self.name,
            "model": self.model_name,
            "initial_task": self.current_task,
        }

    def init(self):
        # Initialize VLM planner
        self.planner = BeliefVLMPlanner(model_name=self.model_name)
        
        # Override system prompt for planning behavior
        self.planner.system_prompt = """You are a robot task planner. Given an image and a specific task instruction, output a JSON object with the following keys:

1. "reasoning": Brief text analysis of the scene relative to the task.
2. "plan": A list of text strings, where each string is a numbered step. 
   Do NOT output objects or bounding boxes. Output ACTION instructions.
3. "belief_update": Text describing what changed in the world state.

Format your response as a valid JSON object:
{
    "reasoning": "I see a cup on the table.",
    "plan": [
        "1. Pick up the cup",
        "2. Place it in the sink"
    ],
    "belief_update": "Cup moved to sink."
}
DO NOT output a list of dictionaries. Output a SINGLE JSON object.
"""
        print(f"[{self.name}] Initialized with model: {self.model_name}")

    def step(self, inp: VLMPlannerInput) -> PlannerOutput:
        if self.planner is None:
            return PlannerOutput(plan=[], success=False, task=self.current_task)
        # Check if we should plan
        current_time = time.time()
        
        # Check for new task (by timestamp)
        task_changed = False
        if inp.task and inp.timestamp > self.last_task_timestamp:
            task_changed = True
            self.current_task = inp.task
            self.last_task_timestamp = inp.timestamp
            self.current_plan = []
            self.last_reasoning = ""
            self.last_belief_update = ""
            self.plan_history = []
             
        # If no task is set, we cannot plan. Return empty/idle.
        if not self.current_task:
             if not self._warned_no_task:
                 print(f"[{self.name}] No task set yet. Use the web UI or --task.")
                 self._warned_no_task = True
             self.current_plan = []
             self.last_reasoning = ""
             self.last_belief_update = ""
             return PlannerOutput(plan=[], success=True, task=self.current_task)

        # Validate inputs
        if inp.state is None:
            # No belief state yet
            if not self._warned_no_state:
                print(f"[{self.name}] Waiting for belief state...")
                self._warned_no_state = True
            return PlannerOutput(
                plan=self.current_plan,
                success=False,
                reasoning=self.last_reasoning,
                belief_update=self.last_belief_update,
                task=self.current_task,
            )
            
        frame_data = self._select_frame(inp)
        if frame_data is None:
            if not self._warned_no_frame:
                print(f"[{self.name}] Waiting for raw observation frame...")
                self._warned_no_frame = True
            return PlannerOutput(
                plan=self.current_plan,
                success=False,
                reasoning=self.last_reasoning,
                belief_update=self.last_belief_update,
                task=self.current_task,
            )
             
        # Replanning check
        should_replan = False
        replan_reason = ""
        
        if task_changed:
            should_replan = True
            replan_reason = "new task"
        elif not self.current_plan:
            should_replan = True
            replan_reason = "no plan"
             
        if not should_replan:
             # Return existing plan or empty if none
             # For simpler logic, we just return empty which implies "no new plan"
             return PlannerOutput(
                 plan=self.current_plan,
                 success=True,
                 reasoning=self.last_reasoning,
                 belief_update=self.last_belief_update,
                 task=self.current_task,
             )
             
        if replan_reason != "new task" and (current_time - self.last_plan_time) < self.min_plan_interval:
            return PlannerOutput(
                plan=self.current_plan,
                success=True,
                reasoning=self.last_reasoning,
                belief_update=self.last_belief_update,
                task=self.current_task,
            )

        print(f"[{self.name}] Planning triggered: {replan_reason} (Task: {self.current_task})")
        
        # Perform VLM Planning
        start = time.time()
        try:
            self.plan_attempt_count += 1
            image = self._load_image(frame_data)
            if image is None:
                if not self._warned_bad_frame:
                    print(f"[{self.name}] Unable to decode frame for VLM planning.")
                    self._warned_bad_frame = True
                return PlannerOutput(
                    plan=self.current_plan,
                    success=False,
                    reasoning=self.last_reasoning,
                    belief_update=self.last_belief_update,
                    task=self.current_task,
                )

            if self._is_black_frame(image):
                if not self._warned_bad_frame:
                    print(f"[{self.name}] Frame appears black; check camera permissions or device index.")
                    self._warned_bad_frame = True
                return PlannerOutput(
                    plan=self.current_plan,
                    success=False,
                    reasoning=self.last_reasoning,
                    belief_update=self.last_belief_update,
                    task=self.current_task,
                )
            
            # Execute VLM Plan
            instruction = self._build_instruction(replan_reason)
            result = self.planner.plan(image, instruction)
            latency_ms = (time.time() - start) * 1000

            if result.get("status") != "success":
                reason = result.get("message") or result.get("reasoning") or "unknown error"
                print(f"[{self.name}] VLM call failed: {reason}")
                return PlannerOutput(
                    plan=self.current_plan,
                    success=False,
                    reasoning=self.last_reasoning,
                    belief_update=self.last_belief_update,
                    task=self.current_task,
                )
            
            parsed = self._parse_plan(result.get("raw_text", ""))

            # Log to Rerun
            rr = self.rr
            if rr:
                if hasattr(rr, "set_time_sequence"):
                    rr.set_time_sequence("plan_step", self.plan_attempt_count)
                else:
                    rr.set_time("plan_step", sequence=self.plan_attempt_count)
                rr.log("metrics/planner_latency_ms", rr.Scalars([latency_ms]))
            
            self.last_plan_time = current_time
            
            # Convert string plan to generic Option/Action if possible, 
            # but for HighLevelPlanner we just output strings in the 'plan' field
            # The FlowTypes.PlannerOutput expects List[Option]. 
            # We might need to wrap strings in a simplified Option or change output type.
            # For now, we return empty Option list, but use a side-channel or specific field if we modify PlannerOutput
            # However, looking at the deprecated demo_highlevel_planning, it used a custom output. 
            # To be compatible with standard flows, we should use standard types.
            # But standard PlannerOutput uses List[Option].
            
            # HACK: For this demo, we will rely on the fact that python is dynamic 
            # and consumers of this flow (VLMExecutionMonitor) will expect strings.
            # Ideally we update the type definition.
            
            plan_strs = self._normalize_plan_steps(parsed.get("plan", []))
            reasoning = parsed.get("reasoning", "")
            belief_update = parsed.get("belief_update", "")
            if plan_strs:
                print(f"[{self.name}] ✓ Generated plan with {len(plan_strs)} steps ({latency_ms:.0f}ms):")
                for i, step in enumerate(plan_strs):
                    print(f"  {i+1}. {step}")

                if reasoning:
                    print(f"[{self.name}] Reasoning: {reasoning[:100]}...")
            else:
                print(f"[{self.name}] ✗ No plan generated")
                print(f"[{self.name}] Raw VLM Output:\n{result.get('raw_text', '')}")
            if plan_strs:
                self.current_plan = plan_strs
            self.last_reasoning = reasoning or self.last_reasoning
            self.last_belief_update = belief_update or self.last_belief_update
            if rr:
                self._log_planner_output(rr, plan_strs, reasoning, belief_update)
            if plan_strs:
                self.plan_history.append(
                    {
                        "attempt": self.plan_attempt_count,
                        "reason": replan_reason,
                        "plan": plan_strs,
                        "reasoning": reasoning,
                        "belief_update": belief_update,
                    }
                )
                if len(self.plan_history) > self.max_plan_history:
                    self.plan_history.pop(0)

            return PlannerOutput(
                plan=plan_strs,
                success=True,
                reasoning=self.last_reasoning,
                belief_update=self.last_belief_update,
                task=self.current_task,
            ) # type: ignore
            
        except Exception as e:
            print(f"[{self.name}] Planning Error: {e}")
            import traceback
            traceback.print_exc()
            return PlannerOutput(
                plan=self.current_plan,
                success=False,
                reasoning=self.last_reasoning,
                belief_update=self.last_belief_update,
                task=self.current_task,
            )

    def _select_frame(self, inp: VLMPlannerInput) -> Optional[Any]:
        if inp.frame is not None:
            return inp.frame

        if inp.state is None:
            return None

        raw_obs = inp.state.raw_observation
        if isinstance(raw_obs, dict):
            for key in ("frame", "rgb"):
                if raw_obs.get(key) is not None:
                    return raw_obs[key]
        elif raw_obs is not None:
            return raw_obs

        data = getattr(inp.state, "data", None)
        if isinstance(data, dict):
            for key in ("frame_bytes", "rgb", "frame"):
                if data.get(key) is not None:
                    return data[key]

        return None

    def _load_image(self, frame: Any):
        from PIL import Image
        import io as iolib

        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, (bytes, bytearray, memoryview)):
            return Image.open(iolib.BytesIO(bytes(frame))).convert("RGB")

        try:
            import numpy as np
        except ImportError:
            return None

        if isinstance(frame, np.ndarray):
            if frame.ndim == 2:
                return Image.fromarray(frame)
            if frame.ndim == 3 and frame.shape[2] >= 3:
                return Image.fromarray(frame[:, :, :3][:, :, ::-1])
            return Image.fromarray(frame)
        return None

    def _is_black_frame(self, image) -> bool:
        extrema = image.getextrema()
        if isinstance(extrema, (list, tuple)) and extrema and isinstance(extrema[0], tuple):
            max_val = max(channel_max for _, channel_max in extrema)
        else:
            max_val = extrema[1] if extrema else 0
        return max_val <= 2

    def _parse_plan(self, raw_text: str) -> dict:
        """Parse VLM response into structured plan (Reused logic)."""
        import json
        try:
            if "```json" in raw_text:
                json_str = raw_text.split("```json")[1].split("```")[0]
            elif "[" in raw_text and ("{" not in raw_text or raw_text.index("[") < raw_text.index("{")):
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
            if isinstance(parsed, list):
                # Check if it looks like detections (list of dicts)
                if parsed and isinstance(parsed[0], dict) and "box_2d" in parsed[0]:
                    # It's detections! The VLM ignored the prompt.
                    return {"reasoning": "VLM output detections instead of plan.", "plan": [], "belief_update": ""}
                
                # Assume list of strings
                return {"reasoning": "", "plan": parsed, "belief_update": ""}
            return parsed
        except (json.JSONDecodeError, ValueError):
            lines = raw_text.split("\n")
            plan = [line.strip() for line in lines if line.strip() and line.strip()[0].isdigit()]
            return {"reasoning": raw_text[:200], "plan": plan, "belief_update": ""}

    def _normalize_plan_steps(self, plan_steps: Any) -> List[str]:
        import re

        if not isinstance(plan_steps, list):
            return []
        cleaned = []
        for step in plan_steps:
            if not isinstance(step, str):
                step = str(step)
            step = re.sub(r"^\\s*\\d+[\\.)]\\s*", "", step).strip()
            if step:
                cleaned.append(step)
        return cleaned

    def _log_planner_output(
        self,
        rr,
        plan_steps: List[str],
        reasoning: str,
        belief_update: str,
    ) -> None:
        lines: List[str] = []
        lines.append("**Plan**")
        if plan_steps:
            for i, step in enumerate(plan_steps, start=1):
                lines.append(f"{i}. {step}")
        else:
            lines.append("(no plan)")
        lines.append("")
        lines.append("**Reasoning**")
        lines.append(reasoning or "(none)")
        lines.append("")
        lines.append("**Belief Update**")
        lines.append(belief_update or "(none)")
        rr.log(
            "planning/planner_output",
            rr.TextDocument("\n".join(lines), media_type="text/markdown"),
        )

    def _build_instruction(self, replan_reason: str) -> str:
        instruction = self.current_task
        history = self._format_plan_history()
        if history:
            instruction = (
                f"{instruction}\n\n"
                "Previous plans (most recent first):\n"
                f"{history}\n\n"
                "If a previous plan is still valid, reuse it. Otherwise update it."
            )
        if replan_reason:
            instruction = f"{instruction}\n\nReplan reason: {replan_reason}"
        return instruction

    def _format_plan_history(self) -> str:
        if not self.plan_history:
            return ""

        lines: List[str] = []
        for entry in reversed(self.plan_history[-3:]):
            attempt = entry.get("attempt", "?")
            reason = entry.get("reason", "unknown")
            lines.append(f"- Attempt {attempt} ({reason})")
            for i, step in enumerate(entry.get("plan", [])):
                lines.append(f"  {i + 1}. {step}")
            reasoning = entry.get("reasoning") or ""
            belief_update = entry.get("belief_update") or ""
            if reasoning:
                lines.append(f"  Reasoning: {reasoning[:160]}")
            if belief_update:
                lines.append(f"  Belief update: {belief_update[:160]}")
        return "\n".join(lines)
