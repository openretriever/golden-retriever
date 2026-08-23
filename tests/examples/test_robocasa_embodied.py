from __future__ import annotations

from types import SimpleNamespace

import pytest
from examples.advanced.robocasa.embodied import (
    EmbodiedGoal,
    GeminiEmbodiedPlanner,
    OfflineEmbodiedPlanner,
    SkillPlan,
    SkillStep,
    _plan_from_payload,
)


def test_offline_planner_builds_dependency_ordered_coffee_plan() -> None:
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(text="Make coffee", task="PrepareCoffee", episode=2)
    )

    assert plan.source == "offline"
    assert plan.goal.text == "Make coffee"
    assert [step.skill for step in plan.steps] == [
        "locate",
        "pick",
        "place",
        "activate",
        "verify",
    ]
    assert plan.steps[1].depends_on == ("step-1",)
    assert plan.step_at(0.5).skill == "place"
    assert plan.step_at(1.0).skill == "verify"


def test_offline_planner_falls_back_to_generic_installed_task() -> None:
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(text="Open the drawer", task="OpenDrawer")
    )

    assert [step.skill for step in plan.steps] == ["execute_demo", "verify"]
    assert plan.goal.task == "OpenDrawer"


def test_skill_plan_rejects_unknown_dependencies_and_skills() -> None:
    goal = EmbodiedGoal(text="Do it", task="PrepareCoffee")

    with pytest.raises(ValueError, match="unsupported skill"):
        SkillPlan(
            goal=goal,
            steps=(SkillStep(step_id="one", skill="shell_command"),),
        ).validate()

    with pytest.raises(ValueError, match="unknown or future"):
        SkillPlan(
            goal=goal,
            steps=(
                SkillStep(
                    step_id="one",
                    skill="pick",
                    depends_on=("missing",),
                ),
            ),
        ).validate()


def test_gemini_planner_accepts_only_allow_listed_structured_steps() -> None:
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **_kwargs: SimpleNamespace(
                text='{"steps":[{"skill":"pick","label":"Pick mug"},'
                '{"skill":"place","label":"Place mug"}]}'
            )
        )
    )
    planner = GeminiEmbodiedPlanner(client=client)

    plan = planner.plan(EmbodiedGoal(text="Move the mug", task="CoffeeSetupMug"))

    assert plan.source == "gemini"
    assert [step.skill for step in plan.steps] == ["pick", "place"]

    with pytest.raises(ValueError, match="unsupported skill"):
        _plan_from_payload(
            EmbodiedGoal(text="Bad plan", task="CoffeeSetupMug"),
            {"steps": [{"skill": "run_python"}]},
            source="gemini",
        )


def test_gemini_without_credentials_uses_offline_fallback(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    plan = GeminiEmbodiedPlanner().plan(
        EmbodiedGoal(text="Make coffee", task="PrepareCoffee", planner="gemini")
    )

    assert plan.source == "offline"
