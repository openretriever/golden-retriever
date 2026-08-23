"""Typed embodied planning contracts for the Retriever RoboCasa console."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from retriever.flow import Flow, io

ALLOWED_SKILLS = frozenset(
    {
        "locate",
        "pick",
        "place",
        "activate",
        "execute_demo",
        "verify",
    }
)


@io
@dataclass(frozen=True)
class EmbodiedGoal:
    """A user goal bound to a simulator task and episode."""

    text: str = ""
    task: str = "TurnOnMicrowave"
    episode: int = 0
    planner: str = "offline"


@io
@dataclass(frozen=True)
class SkillStep:
    """One allow-listed skill in an embodied plan."""

    step_id: str = ""
    skill: str = "execute_demo"
    label: str = "Execute demonstration"
    lane: str = "robot0"
    depends_on: tuple[str, ...] = ()
    start_fraction: float = 0.0
    end_fraction: float = 1.0


@io
@dataclass(frozen=True)
class SkillPlan:
    """A validated, dependency-aware plan for one goal."""

    goal: EmbodiedGoal = field(default_factory=EmbodiedGoal)
    steps: tuple[SkillStep, ...] = ()
    source: str = "offline"

    def validate(self) -> SkillPlan:
        if not self.steps:
            raise ValueError("A skill plan must contain at least one step")
        seen: set[str] = set()
        previous_end = 0.0
        for step in self.steps:
            if not step.step_id or step.step_id in seen:
                raise ValueError(f"Skill step IDs must be unique: {step.step_id!r}")
            if step.skill not in ALLOWED_SKILLS:
                raise ValueError(f"Planner requested unsupported skill: {step.skill}")
            if any(dependency not in seen for dependency in step.depends_on):
                raise ValueError(
                    f"Step {step.step_id!r} depends on an unknown or future step"
                )
            if not 0.0 <= step.start_fraction < step.end_fraction <= 1.0:
                raise ValueError(
                    f"Step {step.step_id!r} has an invalid progress interval"
                )
            if step.start_fraction < previous_end:
                raise ValueError("Skill progress intervals must be ordered")
            seen.add(step.step_id)
            previous_end = step.end_fraction
        return self

    def step_at(self, progress: float) -> SkillStep:
        clamped = min(1.0, max(0.0, float(progress)))
        for step in self.steps:
            if step.start_fraction <= clamped < step.end_fraction:
                return step
        return self.steps[-1]


@io
@dataclass(frozen=True)
class ExecutionEvent:
    """A timestamped lifecycle event emitted by the skill dispatcher."""

    sequence: int = 0
    kind: str = "dispatch"
    status: str = "pending"
    step_id: str = ""
    message: str = ""
    elapsed_seconds: float = 0.0


@io
@dataclass(frozen=True)
class ExecutionState:
    """Current plan and lifecycle state shared with execution Flows."""

    plan: SkillPlan = field(default_factory=SkillPlan)
    status: str = "ready"
    current_step_id: str = ""
    events: tuple[ExecutionEvent, ...] = ()


class Planner(Protocol):
    def plan(self, goal: EmbodiedGoal) -> SkillPlan: ...


@dataclass(frozen=True)
class TaskManifest:
    task: str
    default_goal: str
    aliases: tuple[str, ...]
    steps: tuple[tuple[str, str, float, float], ...]

    def build_plan(self, goal: EmbodiedGoal, *, source: str) -> SkillPlan:
        previous: tuple[str, ...] = ()
        planned: list[SkillStep] = []
        for index, (skill, label, start, end) in enumerate(self.steps, start=1):
            step_id = f"step-{index}"
            planned.append(
                SkillStep(
                    step_id=step_id,
                    skill=skill,
                    label=label,
                    depends_on=previous,
                    start_fraction=start,
                    end_fraction=end,
                )
            )
            previous = (step_id,)
        return SkillPlan(goal=goal, steps=tuple(planned), source=source).validate()


TASK_MANIFESTS: Mapping[str, TaskManifest] = {
    "PrepareCoffee": TaskManifest(
        task="PrepareCoffee",
        default_goal="Prepare a cup of coffee",
        aliases=("prepare coffee", "make coffee", "brew coffee"),
        steps=(
            ("locate", "Locate the mug and coffee machine", 0.00, 0.12),
            ("pick", "Pick the mug from the cabinet", 0.12, 0.48),
            ("place", "Place the mug under the dispenser", 0.48, 0.82),
            ("activate", "Press the coffee machine button", 0.82, 0.96),
            ("verify", "Verify coffee preparation", 0.96, 1.00),
        ),
    ),
    "CoffeeSetupMug": TaskManifest(
        task="CoffeeSetupMug",
        default_goal="Put the mug under the coffee machine",
        aliases=("setup mug", "place mug", "put mug under coffee machine"),
        steps=(
            ("locate", "Locate the mug", 0.00, 0.16),
            ("pick", "Pick the mug from the counter", 0.16, 0.58),
            ("place", "Place the mug under the dispenser", 0.58, 0.92),
            ("verify", "Verify mug placement", 0.92, 1.00),
        ),
    ),
    "StartCoffeeMachine": TaskManifest(
        task="StartCoffeeMachine",
        default_goal="Start the coffee machine",
        aliases=("start coffee", "press coffee button", "turn on coffee machine"),
        steps=(
            ("locate", "Locate the coffee machine control", 0.00, 0.24),
            ("activate", "Press the coffee machine button", 0.24, 0.88),
            ("verify", "Verify the machine is running", 0.88, 1.00),
        ),
    ),
    "TurnOnMicrowave": TaskManifest(
        task="TurnOnMicrowave",
        default_goal="Turn on the microwave",
        aliases=("turn on microwave", "start microwave", "press microwave button"),
        steps=(
            ("locate", "Locate the microwave controls", 0.00, 0.25),
            ("activate", "Press the microwave start button", 0.25, 0.90),
            ("verify", "Verify the microwave is on", 0.90, 1.00),
        ),
    ),
}


class OfflineEmbodiedPlanner:
    """Resolve goals into deterministic, reviewable task manifests."""

    def plan(self, goal: EmbodiedGoal) -> SkillPlan:
        task = goal.task or self._task_from_text(goal.text)
        manifest = TASK_MANIFESTS.get(task)
        if manifest is None:
            manifest = TaskManifest(
                task=task,
                default_goal=f"Run {task}",
                aliases=(),
                steps=(
                    ("execute_demo", f"Execute {task} demonstration", 0.0, 0.94),
                    ("verify", f"Verify {task}", 0.94, 1.0),
                ),
            )
        text = goal.text.strip() or manifest.default_goal
        resolved = EmbodiedGoal(
            text=text,
            task=manifest.task,
            episode=goal.episode,
            planner="offline",
        )
        return manifest.build_plan(resolved, source="offline")

    @staticmethod
    def _task_from_text(text: str) -> str:
        normalized = " ".join(text.lower().split())
        for manifest in TASK_MANIFESTS.values():
            if any(alias in normalized for alias in manifest.aliases):
                return manifest.task
        raise ValueError("Choose a task or enter a supported embodied goal")


class GeminiEmbodiedPlanner:
    """Optional Gemini ER planner restricted to the local skill allow-list."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = "gemini-robotics-er-2-preview",
        fallback: Planner | None = None,
    ) -> None:
        self.model = model
        self.fallback = fallback or OfflineEmbodiedPlanner()
        self._client = client

    def plan(self, goal: EmbodiedGoal) -> SkillPlan:
        if self._client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return self.fallback.plan(goal)
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "Gemini planning requires the optional google-genai package"
                ) from exc
            self._client = genai.Client(api_key=api_key)

        prompt = (
            "Return JSON only. Plan the selected RoboCasa task using the allowed "
            f"skills {sorted(ALLOWED_SKILLS)}. Each step needs skill and label. "
            f"Task: {goal.task}. Goal: {goal.text}"
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        payload = json.loads(_response_text(response))
        return _plan_from_payload(goal, payload, source="gemini")


class GoalSource(Flow[None, EmbodiedGoal]):
    def __init__(self, goal: EmbodiedGoal) -> None:
        self.goal = goal

    def init_config(self) -> dict[str, Any]:
        return {"goal": self.goal.text, "task": self.goal.task}

    def step(self, _input: None = None) -> EmbodiedGoal:
        return self.goal


class EmbodiedPlannerFlow(Flow[EmbodiedGoal, SkillPlan]):
    def __init__(self, planner: Planner) -> None:
        self.planner = planner

    def init_config(self) -> dict[str, Any]:
        return {"planner": type(self.planner).__name__}

    def step(self, goal: EmbodiedGoal) -> SkillPlan:
        return self.planner.plan(goal)


class SkillDispatcher(Flow[SkillPlan, ExecutionState]):
    def __init__(
        self,
        *,
        on_dispatch: Callable[[EmbodiedGoal, SkillPlan], None] | None = None,
    ) -> None:
        self.on_dispatch = on_dispatch
        self._last_signature: tuple[str, str, int] | None = None

    def reset(self) -> None:
        self._last_signature = None

    def step(self, plan: SkillPlan) -> ExecutionState:
        # Retriever supplies @io values as a typed runtime view inside a Flow.
        # Planning validates before dispatch, so only field access belongs here.
        signature = (plan.goal.text, plan.goal.task, plan.goal.episode)
        if signature != self._last_signature:
            if self.on_dispatch is not None:
                self.on_dispatch(plan.goal, plan)
            self._last_signature = signature
        return ExecutionState(
            plan=plan,
            status="ready",
            current_step_id=plan.steps[0].step_id,
        )


def create_planner(name: str, *, client: Any | None = None) -> Planner:
    normalized = name.strip().lower()
    if normalized == "offline":
        return OfflineEmbodiedPlanner()
    if normalized == "gemini":
        return GeminiEmbodiedPlanner(client=client)
    raise ValueError(f"Unknown embodied planner: {name}")


def _plan_from_payload(
    goal: EmbodiedGoal,
    payload: Mapping[str, Any],
    *,
    source: str,
) -> SkillPlan:
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise TypeError("Planner response must contain a steps array")
    if not raw_steps:
        raise ValueError("Planner returned an empty skill plan")

    width = 1.0 / len(raw_steps)
    steps: list[SkillStep] = []
    previous: tuple[str, ...] = ()
    for index, raw in enumerate(raw_steps, start=1):
        if not isinstance(raw, Mapping):
            raise TypeError("Every planner step must be an object")
        skill = str(raw.get("skill", ""))
        if skill not in ALLOWED_SKILLS:
            raise ValueError(f"Planner requested unsupported skill: {skill}")
        step_id = f"step-{index}"
        steps.append(
            SkillStep(
                step_id=step_id,
                skill=skill,
                label=str(raw.get("label") or skill.replace("_", " ").title()),
                depends_on=previous,
                start_fraction=(index - 1) * width,
                end_fraction=index * width,
            )
        )
        previous = (step_id,)
    resolved = EmbodiedGoal(
        text=goal.text,
        task=goal.task,
        episode=goal.episode,
        planner=source,
    )
    return SkillPlan(goal=resolved, steps=tuple(steps), source=source).validate()


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Gemini planner returned no JSON text")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        stripped = stripped.removesuffix("```").strip()
    return stripped
