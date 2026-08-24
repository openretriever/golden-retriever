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
        "open",
        "close",
        "pick",
        "place",
        "activate",
        "execute_demo",
        "verify",
    }
)

EXECUTION_MODES = frozenset({"demonstration", "live_planning"})


@io
@dataclass(frozen=True)
class EmbodiedGoal:
    """A user goal bound to a simulator task and episode."""

    text: str = ""
    task: str = "TurnOnMicrowave"
    episode: int = 0
    planner: str = "offline"
    execution_mode: str = "demonstration"


@io
@dataclass(frozen=True)
class SkillStep:
    """One allow-listed skill in an embodied plan."""

    step_id: str = ""
    skill: str = "execute_demo"
    label: str = "Execute demonstration"
    stage_id: str = "execution"
    stage_label: str = "Execution"
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
        if self.goal.execution_mode not in EXECUTION_MODES:
            raise ValueError(f"Unknown execution mode: {self.goal.execution_mode}")
        seen: set[str] = set()
        closed_stages: set[str] = set()
        active_stage = ""
        previous_end = 0.0
        for step in self.steps:
            if not step.step_id or step.step_id in seen:
                raise ValueError(f"Skill step IDs must be unique: {step.step_id!r}")
            if step.skill not in ALLOWED_SKILLS:
                raise ValueError(f"Planner requested unsupported skill: {step.skill}")
            if not step.stage_id or not step.stage_label:
                raise ValueError("Skill steps require a stage ID and label")
            if step.stage_id != active_stage:
                if step.stage_id in closed_stages:
                    raise ValueError("Skill plan stages must be contiguous")
                if active_stage:
                    closed_stages.add(active_stage)
                active_stage = step.stage_id
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
class ManifestStep:
    skill: str
    label: str
    stage_id: str
    stage_label: str
    start_fraction: float
    end_fraction: float


@dataclass(frozen=True)
class TaskManifest:
    task: str
    default_goal: str
    aliases: tuple[str, ...]
    steps: tuple[ManifestStep, ...]

    def build_plan(self, goal: EmbodiedGoal, *, source: str) -> SkillPlan:
        previous: tuple[str, ...] = ()
        planned: list[SkillStep] = []
        for index, manifest_step in enumerate(self.steps, start=1):
            step_id = f"step-{index}"
            planned.append(
                SkillStep(
                    step_id=step_id,
                    skill=manifest_step.skill,
                    label=manifest_step.label,
                    stage_id=manifest_step.stage_id,
                    stage_label=manifest_step.stage_label,
                    depends_on=previous,
                    start_fraction=manifest_step.start_fraction,
                    end_fraction=manifest_step.end_fraction,
                )
            )
            previous = (step_id,)
        return SkillPlan(goal=goal, steps=tuple(planned), source=source).validate()


TASK_MANIFESTS: Mapping[str, TaskManifest] = {
    "OpenDrawer": TaskManifest(
        task="OpenDrawer",
        default_goal="Open the drawer",
        aliases=("open drawer", "pull open the drawer"),
        steps=(
            ManifestStep("locate", "Locate the drawer handle", "inspect", "Inspect workspace", 0.00, 0.22),
            ManifestStep("open", "Pull the drawer open", "manipulate", "Open drawer", 0.22, 0.92),
            ManifestStep("verify", "Verify the drawer is open", "verify", "Verify outcome", 0.92, 1.00),
        ),
    ),
    "OpenCabinet": TaskManifest(
        task="OpenCabinet",
        default_goal="Open the cabinet",
        aliases=("open cabinet", "open the cupboard"),
        steps=(
            ManifestStep("locate", "Locate the cabinet handle", "inspect", "Inspect workspace", 0.00, 0.20),
            ManifestStep("open", "Swing the cabinet door open", "manipulate", "Open cabinet", 0.20, 0.92),
            ManifestStep("verify", "Verify the cabinet is open", "verify", "Verify outcome", 0.92, 1.00),
        ),
    ),
    "DeliverStraw": TaskManifest(
        task="DeliverStraw",
        default_goal="Take the straw from the drawer and place it in the glass",
        aliases=("deliver straw", "straw from the drawer", "place the straw in the glass"),
        steps=(
            ManifestStep("locate", "Locate the drawer and glass", "inspect", "Inspect workspace", 0.00, 0.10),
            ManifestStep("open", "Open the drawer containing the straw", "retrieve-straw", "Retrieve the straw", 0.10, 0.34),
            ManifestStep("pick", "Pick the straw from the drawer", "retrieve-straw", "Retrieve the straw", 0.34, 0.62),
            ManifestStep("place", "Place the straw inside the glass", "deliver-straw", "Deliver the straw", 0.62, 0.94),
            ManifestStep("verify", "Verify the straw is in the glass", "verify", "Verify outcome", 0.94, 1.00),
        ),
    ),
    "PrepareCoffee": TaskManifest(
        task="PrepareCoffee",
        default_goal="Prepare a cup of coffee",
        aliases=("prepare coffee", "make coffee", "brew coffee"),
        steps=(
            ManifestStep(
                "locate",
                "Locate the mug and coffee machine",
                "inspect",
                "Inspect workspace",
                0.00,
                0.12,
            ),
            ManifestStep(
                "pick",
                "Pick the mug from the cabinet",
                "position-mug",
                "Position the mug",
                0.12,
                0.48,
            ),
            ManifestStep(
                "place",
                "Place the mug under the dispenser",
                "position-mug",
                "Position the mug",
                0.48,
                0.82,
            ),
            ManifestStep(
                "activate",
                "Press the coffee machine button",
                "brew",
                "Brew coffee",
                0.82,
                0.96,
            ),
            ManifestStep(
                "verify",
                "Verify coffee preparation",
                "verify",
                "Verify outcome",
                0.96,
                1.00,
            ),
        ),
    ),
    "LoadDishwasher": TaskManifest(
        task="LoadDishwasher",
        default_goal="Load the cup and bowl into the dishwasher",
        aliases=("load dishwasher", "put dishes in dishwasher"),
        steps=(
            ManifestStep(
                "locate",
                "Locate the cup, bowl, and dishwasher",
                "inspect",
                "Inspect workspace",
                0.00,
                0.10,
            ),
            ManifestStep(
                "pick",
                "Pick the cup from the counter",
                "load-cup",
                "Load the cup",
                0.10,
                0.28,
            ),
            ManifestStep(
                "place",
                "Place the cup on the top rack",
                "load-cup",
                "Load the cup",
                0.28,
                0.44,
            ),
            ManifestStep(
                "pick",
                "Pick the bowl from the counter",
                "load-bowl",
                "Load the bowl",
                0.44,
                0.62,
            ),
            ManifestStep(
                "place",
                "Place the bowl on the top rack",
                "load-bowl",
                "Load the bowl",
                0.62,
                0.78,
            ),
            ManifestStep(
                "activate",
                "Close the dishwasher",
                "finish",
                "Finish loading",
                0.78,
                0.94,
            ),
            ManifestStep(
                "verify",
                "Verify both dishes and the closed door",
                "verify",
                "Verify outcome",
                0.94,
                1.00,
            ),
        ),
    ),
    "LoadFridgeByType": TaskManifest(
        task="LoadFridgeByType",
        default_goal="Load the vegetable and meat bowls onto their assigned fridge shelves",
        aliases=(
            "load fridge by type",
            "sort food into fridge",
            "put food bowls in fridge",
        ),
        steps=(
            ManifestStep(
                "locate",
                "Identify the vegetable bowl, meat bowl, and assigned shelves",
                "inspect",
                "Inspect food and shelves",
                0.00,
                0.10,
            ),
            ManifestStep(
                "pick",
                "Pick the bowl containing vegetables",
                "vegetable-bowl",
                "Store the vegetable bowl",
                0.10,
                0.28,
            ),
            ManifestStep(
                "place",
                "Place the vegetable bowl on its assigned fridge shelf",
                "vegetable-bowl",
                "Store the vegetable bowl",
                0.28,
                0.48,
            ),
            ManifestStep(
                "pick",
                "Pick the bowl containing meat",
                "meat-bowl",
                "Store the meat bowl",
                0.48,
                0.66,
            ),
            ManifestStep(
                "place",
                "Place the meat bowl on its assigned fridge shelf",
                "meat-bowl",
                "Store the meat bowl",
                0.66,
                0.92,
            ),
            ManifestStep(
                "verify",
                "Verify both bowls are on their assigned shelves",
                "verify",
                "Verify outcome",
                0.92,
                1.00,
            ),
        ),
    ),
    "PackIdenticalLunches": TaskManifest(
        task="PackIdenticalLunches",
        default_goal="Pack two identical lunches with vegetables and meat",
        aliases=("pack identical lunches", "pack two lunches"),
        steps=(
            ManifestStep(
                "locate",
                "Locate both containers, vegetables, and meats",
                "inspect",
                "Inspect ingredients",
                0.00,
                0.08,
            ),
            ManifestStep(
                "pick",
                "Pick the first vegetable",
                "lunch-one",
                "Pack the first lunch",
                0.08,
                0.18,
            ),
            ManifestStep(
                "place",
                "Place the vegetable in the first container",
                "lunch-one",
                "Pack the first lunch",
                0.18,
                0.28,
            ),
            ManifestStep(
                "pick",
                "Pick the first meat",
                "lunch-one",
                "Pack the first lunch",
                0.28,
                0.38,
            ),
            ManifestStep(
                "place",
                "Place the meat in the first container",
                "lunch-one",
                "Pack the first lunch",
                0.38,
                0.48,
            ),
            ManifestStep(
                "pick",
                "Pick the second vegetable",
                "lunch-two",
                "Pack the second lunch",
                0.48,
                0.58,
            ),
            ManifestStep(
                "place",
                "Place the vegetable in the second container",
                "lunch-two",
                "Pack the second lunch",
                0.58,
                0.68,
            ),
            ManifestStep(
                "pick",
                "Pick the second meat",
                "lunch-two",
                "Pack the second lunch",
                0.68,
                0.78,
            ),
            ManifestStep(
                "place",
                "Place the meat in the second container",
                "lunch-two",
                "Pack the second lunch",
                0.78,
                0.92,
            ),
            ManifestStep(
                "verify",
                "Verify both lunches have matching contents",
                "verify",
                "Verify outcome",
                0.92,
                1.00,
            ),
        ),
    ),
    "OrganizeCondiments": TaskManifest(
        task="OrganizeCondiments",
        default_goal="Organize the condiments in the cabinet",
        aliases=("organize condiments", "put condiments in cabinet"),
        steps=(
            ManifestStep(
                "locate",
                "Identify the condiments and the distractor",
                "inspect",
                "Inspect objects",
                0.00,
                0.10,
            ),
            ManifestStep(
                "pick",
                "Pick the first condiment",
                "condiment-one",
                "Store condiment one",
                0.10,
                0.22,
            ),
            ManifestStep(
                "place",
                "Place the first condiment in the cabinet",
                "condiment-one",
                "Store condiment one",
                0.22,
                0.34,
            ),
            ManifestStep(
                "pick",
                "Pick the second condiment",
                "condiment-two",
                "Store condiment two",
                0.34,
                0.46,
            ),
            ManifestStep(
                "place",
                "Place the second condiment in the cabinet",
                "condiment-two",
                "Store condiment two",
                0.46,
                0.58,
            ),
            ManifestStep(
                "pick",
                "Pick the third condiment",
                "condiment-three",
                "Store condiment three",
                0.58,
                0.70,
            ),
            ManifestStep(
                "place",
                "Place the third condiment in the cabinet",
                "condiment-three",
                "Store condiment three",
                0.70,
                0.88,
            ),
            ManifestStep(
                "verify",
                "Verify the condiments and untouched distractor",
                "verify",
                "Verify outcome",
                0.88,
                1.00,
            ),
        ),
    ),
    "StackBowlsCabinet": TaskManifest(
        task="StackBowlsCabinet",
        default_goal="Stack the bowls and store them in the cabinet",
        aliases=("stack bowls", "stack bowls in cabinet"),
        steps=(
            ManifestStep(
                "locate",
                "Identify bowl sizes and the target cabinet",
                "inspect",
                "Inspect bowls",
                0.00,
                0.12,
            ),
            ManifestStep(
                "pick",
                "Pick the base bowl",
                "base",
                "Position the base bowl",
                0.12,
                0.28,
            ),
            ManifestStep(
                "place",
                "Place the base bowl in the cabinet",
                "base",
                "Position the base bowl",
                0.28,
                0.48,
            ),
            ManifestStep(
                "pick", "Pick the second bowl", "stack", "Build the stack", 0.48, 0.66
            ),
            ManifestStep(
                "place",
                "Nest the second bowl on the base",
                "stack",
                "Build the stack",
                0.66,
                0.90,
            ),
            ManifestStep(
                "verify",
                "Verify the nested stack is inside the cabinet",
                "verify",
                "Verify outcome",
                0.90,
                1.00,
            ),
        ),
    ),
    "ArrangeDrinkware": TaskManifest(
        task="ArrangeDrinkware",
        default_goal="Arrange the pitcher and cup on the dining counter",
        aliases=(
            "arrange drinkware",
            "set out pitcher and cup",
            "place drinkware for serving",
        ),
        steps=(
            ManifestStep(
                "locate",
                "Locate the pitcher, cup, and dining counter",
                "inspect",
                "Inspect drinkware",
                0.00,
                0.12,
            ),
            ManifestStep(
                "pick",
                "Pick the pitcher from the sink counter",
                "pitcher",
                "Set out the pitcher",
                0.12,
                0.30,
            ),
            ManifestStep(
                "place",
                "Place the pitcher on the dining counter",
                "pitcher",
                "Set out the pitcher",
                0.30,
                0.48,
            ),
            ManifestStep(
                "pick",
                "Pick the cup from the sink counter",
                "cup",
                "Set out the cup",
                0.48,
                0.66,
            ),
            ManifestStep(
                "place",
                "Place the cup on the dining counter",
                "cup",
                "Set out the cup",
                0.66,
                0.92,
            ),
            ManifestStep(
                "verify",
                "Verify the pitcher and cup are ready for serving",
                "verify",
                "Verify outcome",
                0.92,
                1.00,
            ),
        ),
    ),
    "MicrowaveCorrectMeal": TaskManifest(
        task="MicrowaveCorrectMeal",
        default_goal="Microwave the correct meal",
        aliases=("microwave correct meal", "heat the correct meal"),
        steps=(
            ManifestStep(
                "locate",
                "Identify the requested meal bowl",
                "select",
                "Select the meal",
                0.00,
                0.16,
            ),
            ManifestStep(
                "pick",
                "Pick the selected bowl",
                "load",
                "Load the microwave",
                0.16,
                0.38,
            ),
            ManifestStep(
                "place",
                "Place the bowl inside the microwave",
                "load",
                "Load the microwave",
                0.38,
                0.62,
            ),
            ManifestStep(
                "activate",
                "Close the microwave door",
                "cook",
                "Start cooking",
                0.62,
                0.78,
            ),
            ManifestStep(
                "activate",
                "Press the microwave start button",
                "cook",
                "Start cooking",
                0.78,
                0.94,
            ),
            ManifestStep(
                "verify",
                "Verify the selected meal is heating",
                "verify",
                "Verify outcome",
                0.94,
                1.00,
            ),
        ),
    ),
    "PrepareToast": TaskManifest(
        task="PrepareToast",
        default_goal="Prepare bread and jam for toast",
        aliases=("prepare toast", "make toast"),
        steps=(
            ManifestStep(
                "locate",
                "Locate the bread, jam, and cutting board",
                "inspect",
                "Inspect workspace",
                0.00,
                0.10,
            ),
            ManifestStep(
                "pick", "Pick the bread", "stage-bread", "Stage the bread", 0.10, 0.30
            ),
            ManifestStep(
                "place",
                "Place the bread on the cutting board",
                "stage-bread",
                "Stage the bread",
                0.30,
                0.48,
            ),
            ManifestStep(
                "pick", "Pick the jam", "stage-jam", "Stage the jam", 0.48, 0.66
            ),
            ManifestStep(
                "place",
                "Place the jam beside the cutting board",
                "stage-jam",
                "Stage the jam",
                0.66,
                0.82,
            ),
            ManifestStep(
                "activate", "Close the cabinet", "finish", "Finish setup", 0.82, 0.94
            ),
            ManifestStep(
                "verify",
                "Verify the toast ingredients are ready",
                "verify",
                "Verify outcome",
                0.94,
                1.00,
            ),
        ),
    ),
    "RestockPantry": TaskManifest(
        task="RestockPantry",
        default_goal="Restock the pantry cans on the matching side",
        aliases=("restock pantry", "put cans in pantry"),
        steps=(
            ManifestStep(
                "locate",
                "Identify the matching pantry side",
                "inspect",
                "Match the category",
                0.00,
                0.14,
            ),
            ManifestStep(
                "pick",
                "Pick the first can",
                "first-can",
                "Restock the first can",
                0.14,
                0.32,
            ),
            ManifestStep(
                "place",
                "Place the first can on the matching side",
                "first-can",
                "Restock the first can",
                0.32,
                0.48,
            ),
            ManifestStep(
                "pick",
                "Pick the second can",
                "second-can",
                "Restock the second can",
                0.48,
                0.66,
            ),
            ManifestStep(
                "place",
                "Place the second can on the matching side",
                "second-can",
                "Restock the second can",
                0.66,
                0.92,
            ),
            ManifestStep(
                "verify",
                "Verify both cans are correctly grouped",
                "verify",
                "Verify outcome",
                0.92,
                1.00,
            ),
        ),
    ),
    "SetupFrying": TaskManifest(
        task="SetupFrying",
        default_goal="Set up the pan and burner for frying",
        aliases=("setup frying", "set up frying", "prepare pan for frying"),
        steps=(
            ManifestStep(
                "locate",
                "Locate the pan and target burner",
                "retrieve",
                "Retrieve cookware",
                0.00,
                0.14,
            ),
            ManifestStep(
                "pick",
                "Pick the pan from the cabinet",
                "retrieve",
                "Retrieve cookware",
                0.14,
                0.42,
            ),
            ManifestStep(
                "place",
                "Place the pan on the target burner",
                "position",
                "Position cookware",
                0.42,
                0.78,
            ),
            ManifestStep(
                "activate",
                "Turn on the matching burner",
                "heat",
                "Start heat",
                0.78,
                0.94,
            ),
            ManifestStep(
                "verify",
                "Verify the pan and burner state",
                "verify",
                "Verify outcome",
                0.94,
                1.00,
            ),
        ),
    ),
    "CoffeeSetupMug": TaskManifest(
        task="CoffeeSetupMug",
        default_goal="Put the mug under the coffee machine",
        aliases=("setup mug", "place mug", "put mug under coffee machine"),
        steps=(
            ManifestStep(
                "locate", "Locate the mug", "inspect", "Inspect workspace", 0.00, 0.16
            ),
            ManifestStep(
                "pick",
                "Pick the mug from the counter",
                "move-mug",
                "Move the mug",
                0.16,
                0.58,
            ),
            ManifestStep(
                "place",
                "Place the mug under the dispenser",
                "move-mug",
                "Move the mug",
                0.58,
                0.92,
            ),
            ManifestStep(
                "verify", "Verify mug placement", "verify", "Verify outcome", 0.92, 1.00
            ),
        ),
    ),
    "StartCoffeeMachine": TaskManifest(
        task="StartCoffeeMachine",
        default_goal="Start the coffee machine",
        aliases=("start coffee", "press coffee button", "turn on coffee machine"),
        steps=(
            ManifestStep(
                "locate",
                "Locate the coffee machine control",
                "inspect",
                "Inspect controls",
                0.00,
                0.24,
            ),
            ManifestStep(
                "activate",
                "Press the coffee machine button",
                "activate",
                "Start the machine",
                0.24,
                0.88,
            ),
            ManifestStep(
                "verify",
                "Verify the machine is running",
                "verify",
                "Verify outcome",
                0.88,
                1.00,
            ),
        ),
    ),
    "TurnOnMicrowave": TaskManifest(
        task="TurnOnMicrowave",
        default_goal="Turn on the microwave",
        aliases=("turn on microwave", "start microwave", "press microwave button"),
        steps=(
            ManifestStep(
                "locate",
                "Locate the microwave controls",
                "inspect",
                "Inspect controls",
                0.00,
                0.25,
            ),
            ManifestStep(
                "activate",
                "Press the microwave start button",
                "activate",
                "Start the microwave",
                0.25,
                0.90,
            ),
            ManifestStep(
                "verify",
                "Verify the microwave is on",
                "verify",
                "Verify outcome",
                0.90,
                1.00,
            ),
        ),
    ),
}


class OfflineEmbodiedPlanner:
    """Resolve goals into deterministic, reviewable task manifests."""

    def plan(self, goal: EmbodiedGoal) -> SkillPlan:
        if goal.execution_mode not in EXECUTION_MODES:
            raise ValueError(f"Unknown execution mode: {goal.execution_mode}")
        task = goal.task or self._task_from_text(goal.text)
        manifest = TASK_MANIFESTS.get(task)
        if manifest is None:
            manifest = TaskManifest(
                task=task,
                default_goal=f"Run {task}",
                aliases=(),
                steps=(
                    ManifestStep(
                        "execute_demo",
                        f"Execute {task} demonstration",
                        "execution",
                        "Execute demonstration",
                        0.0,
                        0.94,
                    ),
                    ManifestStep(
                        "verify",
                        f"Verify {task}",
                        "verify",
                        "Verify outcome",
                        0.94,
                        1.0,
                    ),
                ),
            )
        text = goal.text.strip() or manifest.default_goal
        resolved = EmbodiedGoal(
            text=text,
            task=manifest.task,
            episode=goal.episode,
            planner="offline",
            execution_mode=goal.execution_mode,
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
                stage_id=str(raw.get("stage") or raw.get("stage_id") or "execution"),
                stage_label=str(raw.get("stage_label") or "Execution"),
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
        execution_mode=goal.execution_mode,
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
