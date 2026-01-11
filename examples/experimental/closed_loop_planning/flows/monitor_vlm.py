from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import time
import sys
from retriever.flow import Flow, flow_io
from retriever.types.options import Option
from ..types.belief import BeliefState
from ..types.flow_types import MonitorOutput, ExecutionState
from ..pipelines.vlm_utils import BeliefVLMPlanner
import os

@flow_io
@dataclass
class VLMMonitorInput:
    state: BeliefState = None  # type: ignore[assignment]
    frame: Optional[bytes] = None
    plan: List[Option] = field(default_factory=list)
    task: str = ""
    timestamp: float = 0.0
    reasoning: str = ""
    belief_update: str = ""


class VLMExecutionMonitorFlow(Flow[VLMMonitorInput, MonitorOutput]):
    """
    VLM-based execution monitor that checks if current instruction is complete.
    
    Uses VLM to analyze webcam feed and determine if the current step
    in the plan has been accomplished visually.
    """
    
    def __init__(self, name: str = "VLMMonitor", model: str = "gemini-2.5-flash-lite"):
        self.name = name
        # Allow env var override since dora might re-instantiate with default args
        self.model_name = os.getenv("VLM_MODEL", model)
        self.vlm: Optional[BeliefVLMPlanner] = None
        self.state_vlm: Optional[BeliefVLMPlanner] = None
        self.state = ExecutionState.IDLE
        self.current_plan: List[str] = []
        self.current_step = 0
        self.current_task = ""
        self.current_reasoning = ""
        self.current_belief_update = ""
        self.current_state_desc = ""
        self.start_state_desc = ""
        self.goal_state_desc = ""
        self.last_check_status = "N/A"
        self.last_task_check_status = "N/A"
        self.last_task_timestamp = 0.0
        self.task_completed = False
        self.last_task_check_time = 0.0
        self.min_task_check_interval = 3.0
        self.last_state_guess_time = 0.0
        self.min_state_guess_interval = 5.0
        self._last_summary_text = ""
        self._last_plan_html = ""
        self._plan_html_path: Optional[Path] = None
        self._last_plan_html_opened = ""

    def init_config(self) -> dict:
        return {"name": self.name, "model": self.model_name}
    
    def init(self):
        # Initialize VLM for task completion checking
        self.vlm = BeliefVLMPlanner(model_name=self.model_name)
        self.vlm.system_prompt = """You are a visual task completion checker.
Given an image and a task instruction, determine if the task appears to be complete.

Respond with ONLY one of:
- "COMPLETE" if the task is visibly finished
- "IN_PROGRESS" if the task is being executed but not done
- "NOT_STARTED" if there's no evidence the task has begun

Be conservative - only say COMPLETE if you're confident."""

        self.state_vlm = BeliefVLMPlanner(model_name=self.model_name)
        self.state_vlm.system_prompt = """You are a world-state summarizer.
Given an image and the task context, describe the current state in ONE sentence.
Focus on objects, their locations, and any progress toward the task.
Output plain text only."""
        
        print(f"[{self.name}] Initialized VLM monitor with {self.model_name}")

        rr = self.rr
        if rr:
            try:
                import rerun.blueprint as rrb

                layout = rrb.Horizontal(
                    rrb.Vertical(
                        rrb.Spatial2DView(
                            origin="/camera/webcam",
                            contents="/camera/webcam",
                            name="Camera",
                        ),
                        rrb.TimeSeriesView(
                            origin="/metrics",
                            contents="/metrics/planner_latency_ms",
                            name="Planner Latency",
                            time_ranges=[
                                rrb.VisibleTimeRange(
                                    "plan_step",
                                    start=rrb.TimeRangeBoundary.absolute(seq=0),
                                    end=rrb.TimeRangeBoundary.infinite(),
                                )
                            ],
                        ),
                        rrb.TextLogView(
                            origin="/perception/status",
                            contents="/perception/status",
                            name="Perception Output",
                        ),
                        rrb.TextLogView(
                            origin="/belief/status",
                            contents="/belief/status",
                            name="Belief Output",
                        ),
                    ),
                    rrb.Vertical(
                        rrb.TextDocumentView(
                            origin="/planning/summary",
                            contents="/planning/summary",
                            name="Planning Summary",
                        ),
                    ),
                )

                rr.send_blueprint(
                    rrb.Blueprint(
                        layout,
                        rrb.TimePanel(timeline="log_time"),
                        auto_layout=False,
                        auto_views=False,
                    )
                )
            except Exception:
                pass
    
    def step(self, inp: VLMMonitorInput) -> MonitorOutput:
        now = time.time()
        incoming_plan = [step if isinstance(step, str) else str(step) for step in inp.plan]

        # Update plan if received
        if incoming_plan and incoming_plan != self.current_plan:
            self.current_plan = incoming_plan
            self.current_step = 0
            self.state = ExecutionState.EXECUTING
            self.last_check_status = "PENDING"
            self.task_completed = False
            self.last_task_check_status = "PENDING"
            if self.current_state_desc and not self.start_state_desc:
                self.start_state_desc = self.current_state_desc
            print(f"[{self.name}] New plan with {len(incoming_plan)} steps")

        if inp.task and inp.timestamp > self.last_task_timestamp:
            self.current_task = inp.task
            self.last_task_timestamp = inp.timestamp
            self.current_plan = []
            self.current_step = 0
            self.state = ExecutionState.PLANNING
            self.task_completed = False
            self.last_task_check_status = "PENDING"
            self.goal_state_desc = self._goal_state_from_task(self.current_task)
            self.start_state_desc = ""
        elif inp.task:
            self.current_task = inp.task
        if inp.reasoning:
            self.current_reasoning = inp.reasoning
        if inp.belief_update:
            self.current_belief_update = inp.belief_update
        
        frame = inp.frame
        if frame is None:
             # Try to get from belief if available?
             if inp.state and inp.state.raw_observation:
                  frame = inp.state.raw_observation.get("frame")

        # If we have a plan and frame, check if current step is complete
        if (
            self.current_plan
            and self.current_step < len(self.current_plan)
            and frame is not None
            and self.state == ExecutionState.EXECUTING
        ):
            current_instruction = self.current_plan[self.current_step]
            
            # Check completion using VLM
            try:
                from PIL import Image
                import io as iolib
                image = Image.open(iolib.BytesIO(frame))
                
                check_prompt = f"Current instruction: {current_instruction}"
                result = self.vlm.plan(image, check_prompt)
                
                if result.get("status") == "success":
                    response = result.get("raw_text", "")
                    token = response.strip().splitlines()[0] if response else ""
                    token = token.strip().strip("\"'.,").upper()
                    token = token.replace(" ", "_").replace("-", "_")

                    if token == "COMPLETE":
                        self.last_check_status = "COMPLETE"
                        print(f"[{self.name}] ✓ Step {self.current_step + 1} COMPLETE: {current_instruction}")
                        self.current_step += 1
                        
                        # Check if all steps done
                        if self.current_step >= len(self.current_plan):
                            self.state = ExecutionState.IDLE
                            print(f"[{self.name}] ✓✓ All steps complete!")
                    elif token == "IN_PROGRESS":
                        self.last_check_status = "IN_PROGRESS"
                        print(f"[{self.name}] ... Step {self.current_step + 1} in progress")
                    elif token == "NOT_STARTED":
                        self.last_check_status = "NOT_STARTED"
                    else:
                        self.last_check_status = "UNKNOWN"
                else:
                    self.last_check_status = "ERROR"
                        
            except Exception as e:
                self.last_check_status = "ERROR"
                print(f"[{self.name}] VLM check error: {e}")
        elif self.current_plan and frame is None:
            self.last_check_status = "NO_FRAME"
        elif not self.current_plan:
            self.last_check_status = "NO_PLAN"

        replan_config = None

        if frame is not None and (now - self.last_state_guess_time) >= self.min_state_guess_interval:
            self.last_state_guess_time = now
            state_guess = self._guess_current_state(frame)
            if state_guess:
                self.current_state_desc = state_guess
                if self.current_plan and not self.start_state_desc:
                    self.start_state_desc = state_guess

        if (
            self.current_plan
            and self.current_step >= len(self.current_plan)
            and frame is not None
            and (now - self.last_task_check_time) >= self.min_task_check_interval
        ):
            self.last_task_check_time = now
            self.last_task_check_status = self._check_task_completion(frame)
            if self.last_task_check_status == "COMPLETE":
                self.task_completed = True

        # Build Markdown summary
        task_status = self._task_status()
        summary_lines = []
        plan_len = len(self.current_plan)
        if plan_len:
            step_display = f"{min(self.current_step + 1, plan_len)}/{plan_len}"
        else:
            step_display = "0/0"
        summary_lines.append(
            f"**Task Status**: {task_status}"
        )
        summary_lines.append(
            f"**Execution**: {self.state.value} | Step {step_display} | Check: {self.last_check_status}"
        )
        summary_lines.append(f"**Task**: {self.current_task or '(none)'}")

        current_step_str = (
            self.current_plan[self.current_step]
            if self.current_step < len(self.current_plan)
            else "(none)"
        )
        summary_lines.append(f"**Current Step**: {current_step_str}")
        if self.current_state_desc:
            summary_lines.append(f"**Current State**: {self.current_state_desc}")
        if self.current_plan and self.current_step >= len(self.current_plan):
            summary_lines.append(f"**Task Check**: {self.last_task_check_status}")

        summary_lines.append("")
        summary_lines.append("**Plan (next steps)**")
        if self.current_plan:
            window = 5
            start = self.current_step
            end = min(len(self.current_plan), start + window)
            for i in range(start, end):
                marker = "->" if i == self.current_step else "  "
                summary_lines.append(f"{marker} {i + 1}. {self.current_plan[i]}")
        else:
            summary_lines.append("(no plan)")

        summary_lines.append("")
        plan_tree_text = self._render_plan_tree_text()
        summary_lines.extend(plan_tree_text.splitlines())

        plan_html_path = self._write_plan_html()
        if plan_html_path:
            summary_lines.append("")
            summary_lines.append("**Plan Graph (HTML)**")
            summary_lines.append(f"Path: `{plan_html_path}`")
            summary_lines.append(f"Open: `open {plan_html_path}`")

        summary_lines.append("")
        summary_lines.append("**Monitor Output**")
        summary_lines.extend(
            [
                f"Execution State: {self.state.value}",
                f"Step: {step_display}",
                f"Check: {self.last_check_status}",
                f"Task Check: {self.last_task_check_status}",
                f"State Guess: {self.current_state_desc or '(none)'}",
            ]
        )

        summary_lines.append("")
        summary_lines.append("**Reasoning**")
        summary_lines.append(self.current_reasoning or "(none)")
        summary_lines.append("")
        summary_lines.append("**Belief Update**")
        summary_lines.append(self.current_belief_update or "(none)")

        summary_text = "\n".join(summary_lines)

        # Log to Rerun
        rr = self.rr
        if rr:
            if summary_text != self._last_summary_text:
                self._last_summary_text = summary_text
                rr.log("planning/summary", rr.TextDocument(summary_text, media_type="text/markdown"))
            if plan_html_path:
                self._maybe_open_plan_html(plan_html_path)
        
        # We don't log reasoning/belief here as that belongs to Planner/BeliefUpdater
        
        current_step_str = self.current_plan[self.current_step] if self.current_step < len(self.current_plan) else ""
        
        return MonitorOutput(
            replan_config=replan_config,
            state=self.state,
            current_step=self.current_step,
            current_instruction=current_step_str,
            plan_display=summary_text,
            task_completed=self.task_completed,
            task_status=task_status,
        )

    def _task_status(self) -> str:
        if self.task_completed:
            return "COMPLETED"
        if not self.current_task:
            return "NO_TASK"
        if not self.current_plan:
            return "NO_PLAN"
        if self.current_step >= len(self.current_plan):
            if self.last_task_check_status in ("IN_PROGRESS", "NOT_STARTED"):
                return "INCOMPLETE"
            return "VERIFYING"
        return "IN_PROGRESS"

    def _check_task_completion(self, frame: bytes) -> str:
        if self.vlm is None:
            return "UNKNOWN"
        try:
            from PIL import Image
            import io as iolib

            image = Image.open(iolib.BytesIO(frame))
            check_prompt = f"Overall task: {self.current_task}"
            result = self.vlm.plan(image, check_prompt)
            if result.get("status") != "success":
                return "ERROR"
            response = result.get("raw_text", "")
            token = response.strip().splitlines()[0] if response else ""
            token = token.strip().strip("\"'.,").upper()
            token = token.replace(" ", "_").replace("-", "_")
            return token or "UNKNOWN"
        except Exception:
            return "ERROR"

    def _render_plan_tree_text(self) -> str:
        lines = ["**Plan Tree**"]
        lines.extend(self._build_plan_tree_lines(markdown=True))
        return "\n".join(lines)

    def _maybe_open_plan_html(self, plan_html_path: str) -> None:
        auto_open = os.getenv("RETRIEVER_PLAN_HTML_AUTO_OPEN", "1").strip().lower()
        if auto_open in ("0", "false", "no", "off"):
            return
        if not plan_html_path or plan_html_path == self._last_plan_html_opened:
            return

        cmd = None
        if sys.platform == "darwin":
            cmd = ["open", plan_html_path]
        elif sys.platform.startswith("linux"):
            cmd = ["xdg-open", plan_html_path]
        elif os.name == "nt":
            cmd = ["cmd", "/c", "start", "", plan_html_path]

        if not cmd:
            return

        self._last_plan_html_opened = plan_html_path
        try:
            import subprocess

            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=os.name != "nt",
            )
        except Exception:
            pass

    def _state_description(self, idx: int, state_count: int, current_state_idx: int) -> str:
        if idx == 0:
            return self.start_state_desc or "Start state"
        if idx == state_count - 1:
            return self.goal_state_desc or "Goal state"
        if idx == current_state_idx:
            return self.current_state_desc or "Current state"
        return "State"

    def _state_label(
        self,
        idx: int,
        state_count: int,
        current_state_idx: int,
        include_index: bool = True,
    ) -> str:
        desc = self._state_description(idx, state_count, current_state_idx)
        if include_index:
            return f"[{idx + 1}] {desc}" if desc else f"[{idx + 1}]"
        return desc

    def _build_plan_tree_lines(self, markdown: bool = False) -> List[str]:
        state_count = len(self.current_plan) + 1 if self.current_plan else 0
        current_state_idx = (
            min(self.current_step, state_count - 1) if state_count else 0
        )

        if not self.current_plan:
            lines = ["(no plan)"]
            if self.current_task:
                lines.append(f"Task: {self.current_task}")
            if markdown:
                return ["```", *lines, "```"]
            return lines

        def state_tags(idx: int) -> str:
            tags = []
            if idx == 0:
                tags.append("start")
            if idx == state_count - 1:
                tags.append("goal")
            if idx == current_state_idx and self.state == ExecutionState.EXECUTING:
                tags.append("current")
            return f" ({', '.join(tags)})" if tags else ""

        def state_marker(idx: int) -> str:
            if idx == current_state_idx and self.state == ExecutionState.EXECUTING:
                return " [<-]"
            return ""

        def action_marker(idx: int) -> str:
            if idx < self.current_step:
                return "[x]"
            if idx == self.current_step and self.state == ExecutionState.EXECUTING:
                return "[>>]"
            return "[ ]"

        def state_hier_id(idx: int) -> str:
            return "S1" if idx == 0 else f"S1.{idx}"

        def action_hier_id(idx: int) -> str:
            return f"A1.{idx + 1}"

        lines: List[str] = []

        def add_state(idx: int, indent: str, root: bool = False) -> None:
            desc = self._truncate_text(
                self._state_description(idx, state_count, current_state_idx), 120
            )
            line = (
                f"{state_hier_id(idx)} [{idx + 1}]"
                f"{state_tags(idx)}{state_marker(idx)} {desc}"
            ).rstrip()
            prefix = "" if root else "└─ "
            lines.append(f"{indent}{prefix}{line}")

            if idx < len(self.current_plan):
                action_text = self._truncate_text(self.current_plan[idx], 140)
                action_line = (
                    f"{action_hier_id(idx)} (step {idx + 1}) "
                    f"{action_marker(idx)} {action_text}"
                ).rstrip()
                action_indent = indent + ("   " if root else "   ")
                lines.append(f"{action_indent}└─ {action_line}")
                add_state(idx + 1, action_indent + "   ")

        add_state(0, "", root=True)

        if markdown:
            return ["```", *lines, "```"]
        return lines

    def _state_color(self, idx: int, state_count: int, current_state_idx: int):
        start_color = (52, 152, 219, 255)
        goal_color = (155, 89, 182, 255)
        current_color = (241, 196, 15, 255)
        default_color = (180, 180, 180, 255)

        if idx == 0:
            return start_color
        if idx == state_count - 1:
            return goal_color
        if idx == current_state_idx and self.state == ExecutionState.EXECUTING:
            return current_color
        return default_color

    def _goal_state_from_task(self, task: str) -> str:
        if not task:
            return "Goal state"
        return f"Goal: {task}"

    def _guess_current_state(self, frame: bytes) -> str:
        if self.state_vlm is None:
            return ""
        try:
            from PIL import Image
            import io as iolib

            image = Image.open(iolib.BytesIO(frame))
            prompt = "Describe the current state of the scene in one sentence."
            if self.current_task:
                prompt = f"{prompt} Task: {self.current_task}"
            result = self.state_vlm.plan(image, prompt)
            if result.get("status") != "success":
                return ""
            text = result.get("reasoning") or result.get("raw_text") or ""
            return text.strip().splitlines()[0] if text else ""
        except Exception:
            return ""

    def _truncate_text(self, text: str, limit: int) -> str:
        cleaned = text.strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 3)].rstrip() + "..."

    def _write_plan_html(self) -> Optional[str]:
        if not self.current_plan:
            return None

        import json
        import html

        state_count = len(self.current_plan) + 1
        current_state_idx = min(self.current_step, state_count - 1)
        nodes = []
        edges = []

        def state_hier_id(idx: int) -> str:
            return "S1" if idx == 0 else f"S1.{idx}"

        def action_hier_id(idx: int) -> str:
            return f"A1.{idx + 1}"

        for i in range(state_count):
            node_id = f"s{i}"
            label = f"[{i + 1}]"
            classes = ["state"]
            if i == 0:
                classes.append("start")
            if i == state_count - 1:
                classes.append("goal")
            if i == current_state_idx and self.state == ExecutionState.EXECUTING:
                classes.append("current")
            nodes.append({"data": {"id": node_id, "label": label}, "classes": " ".join(classes)})

        for i, action in enumerate(self.current_plan):
            edge_id = f"e{i}"
            classes = ["edge"]
            if i < self.current_step:
                classes.append("done")
            elif i == self.current_step:
                classes.append("current")
            edges.append(
                {
                    "data": {
                        "id": edge_id,
                        "source": f"s{i}",
                        "target": f"s{i + 1}",
                        "label": action_hier_id(i),
                    },
                    "classes": " ".join(classes),
                }
            )

        elements = {"nodes": nodes, "edges": edges}
        elements_json = json.dumps(elements, ensure_ascii=True)
        title = self.current_task or "Plan Graph"

        state_items = []
        for i in range(state_count):
            desc = self._state_description(i, state_count, current_state_idx)
            desc = desc if desc else "State"
            tags = []
            classes = ["state-item"]
            if i == 0:
                tags.append("start")
                classes.append("start")
            if i == state_count - 1:
                tags.append("goal")
                classes.append("goal")
            if i == current_state_idx and self.state == ExecutionState.EXECUTING:
                tags.append("current")
                classes.append("current")
            tag_text = f" ({', '.join(tags)})" if tags else ""
            state_items.append(
                "<li class=\"{cls}\">"
                "<span class=\"id\">{sid}</span> "
                "<span class=\"num\">[{num}]</span>"
                "<span class=\"tag\">{tags}</span>"
                "<div class=\"desc\">{desc}</div>"
                "</li>".format(
                    cls=" ".join(classes),
                    sid=html.escape(state_hier_id(i)),
                    num=i + 1,
                    tags=html.escape(tag_text),
                    desc=html.escape(desc),
                )
            )

        action_items = []
        for i, action in enumerate(self.current_plan):
            status = "done" if i < self.current_step else "current" if i == self.current_step else "pending"
            marker = "[x]" if status == "done" else "[>>]" if status == "current" else "[ ]"
            action_items.append(
                "<li class=\"action-item {status}\">"
                "<span class=\"id\">{aid}</span> "
                "<span class=\"step\">(step {step})</span> "
                "<span class=\"marker\">{marker}</span>"
                "<div class=\"desc\">{desc}</div>"
                "</li>".format(
                    status=status,
                    aid=html.escape(action_hier_id(i)),
                    step=i + 1,
                    marker=html.escape(marker),
                    desc=html.escape(action),
                )
            )

        tree_text = "\n".join(self._build_plan_tree_lines(markdown=False))
        current_state_desc = self._state_description(current_state_idx, state_count, current_state_idx)
        current_action_desc = (
            self.current_plan[self.current_step]
            if self.current_step < len(self.current_plan)
            else "(none)"
        )
        current_state_label = f"{state_hier_id(current_state_idx)} [{current_state_idx + 1}]"
        current_action_label = (
            f"{action_hier_id(self.current_step)} (step {self.current_step + 1})"
            if self.current_step < len(self.current_plan)
            else "(none)"
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    html, body {{
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      background: #ffffff;
      font-family: Arial, sans-serif;
      color: #222;
    }}
    #container {{
      display: flex;
      height: 100%;
      width: 100%;
    }}
    #cy {{
      flex: 1;
      min-width: 0;
      height: 100%;
    }}
    #panel {{
      width: 340px;
      background: #fafafa;
      border-left: 1px solid #e0e0e0;
      padding: 12px;
      box-sizing: border-box;
      overflow-y: auto;
    }}
    .panel-title {{
      font-size: 14px;
      font-weight: 600;
      margin: 0 0 6px 0;
    }}
    .section {{
      margin-bottom: 14px;
    }}
    .section h3 {{
      font-size: 12px;
      margin: 10px 0 6px 0;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #555;
    }}
    .kv {{
      font-size: 12px;
      margin: 4px 0;
    }}
    .kv span {{
      display: inline-block;
      min-width: 64px;
      font-weight: 600;
    }}
    ul {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    li {{
      margin-bottom: 8px;
      padding-bottom: 6px;
      border-bottom: 1px solid #eee;
    }}
    .id {{
      font-weight: 700;
    }}
    .num, .step {{
      color: #777;
      margin-left: 4px;
    }}
    .tag {{
      color: #888;
      margin-left: 6px;
      font-size: 11px;
    }}
    .desc {{
      margin-top: 4px;
      font-size: 12px;
      color: #333;
    }}
    .state-item.start .id {{ color: #3498db; }}
    .state-item.goal .id {{ color: #2ecc71; }}
    .state-item.current .id {{ color: #e67e22; }}
    .action-item.current .id {{ color: #e67e22; }}
    .action-item.done .id {{ color: #3498db; }}
    .marker {{
      margin-left: 6px;
      font-weight: 600;
      color: #444;
    }}
    pre.tree {{
      background: #fff;
      border: 1px solid #e0e0e0;
      padding: 8px;
      font-size: 11px;
      white-space: pre-wrap;
      margin: 0;
    }}
  </style>
  <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
</head>
<body>
  <div id="container">
    <div id="cy"></div>
    <div id="panel">
      <div class="section">
        <div class="panel-title">Plan Graph</div>
        <div class="kv"><span>Task</span>{html.escape(title)}</div>
      </div>
      <div class="section">
        <h3>Current</h3>
        <div class="kv"><span>State</span>{html.escape(current_state_label)}</div>
        <div class="desc">{html.escape(current_state_desc)}</div>
        <div class="kv"><span>Action</span>{html.escape(current_action_label)}</div>
        <div class="desc">{html.escape(current_action_desc)}</div>
      </div>
      <div class="section">
        <h3>States</h3>
        <ul>
          {"".join(state_items)}
        </ul>
      </div>
      <div class="section">
        <h3>Actions</h3>
        <ul>
          {"".join(action_items)}
        </ul>
      </div>
      <div class="section">
        <h3>Tree</h3>
        <pre class="tree">{html.escape(tree_text)}</pre>
      </div>
    </div>
  </div>
  <script>
    if (!window.cytoscape) {{
      document.getElementById('cy').innerText = 'Cytoscape failed to load.';
    }} else {{
      var elements = {elements_json};
      var cy = cytoscape({{
        container: document.getElementById('cy'),
        elements: elements,
        layout: {{
          name: 'breadthfirst',
          directed: true,
          spacingFactor: 1.6,
          animate: false
        }},
        style: [
          {{
            selector: 'node.state',
            style: {{
              'shape': 'ellipse',
              'background-color': '#ffffff',
              'border-color': '#333',
              'border-width': 1,
              'label': 'data(label)',
              'text-wrap': 'none',
              'font-size': 12,
              'text-valign': 'center',
              'text-halign': 'center'
            }}
          }},
          {{
            selector: 'node.start',
            style: {{
              'border-color': '#3498db',
              'border-width': 3
            }}
          }},
          {{
            selector: 'node.goal',
            style: {{
              'border-color': '#2ecc71',
              'border-width': 3
            }}
          }},
          {{
            selector: 'node.current',
            style: {{
              'border-color': '#f1c40f',
              'border-width': 3,
              'background-color': '#fffbe6'
            }}
          }},
          {{
            selector: 'edge',
            style: {{
              'curve-style': 'bezier',
              'target-arrow-shape': 'triangle',
              'line-color': '#777',
              'target-arrow-color': '#777',
              'width': 2,
              'label': 'data(label)',
              'font-size': 10,
              'text-rotation': 'autorotate',
              'text-margin-y': -6,
              'text-background-color': '#ffffff',
              'text-background-opacity': 0.9,
              'text-background-padding': 2
            }}
          }},
          {{
            selector: 'edge.done',
            style: {{
              'line-color': '#3498db',
              'target-arrow-color': '#3498db'
            }}
          }},
          {{
            selector: 'edge.current',
            style: {{
              'line-color': '#e67e22',
              'target-arrow-color': '#e67e22'
            }}
          }}
        ]
      }});
      cy.fit();
    }}
  </script>
</body>
</html>"""

        if html == self._last_plan_html and self._plan_html_path:
            return str(self._plan_html_path)

        output_dir = Path(os.getenv("RETRIEVER_PLAN_HTML_DIR", "./logs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"plan_graph_{os.getpid()}.html"
        plan_path = output_dir / file_name
        plan_path.write_text(html, encoding="utf-8")
        self._plan_html_path = plan_path
        self._last_plan_html = html
        return str(plan_path)
