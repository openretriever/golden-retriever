from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples.advanced.robocasa.embodied import (
    TASK_MANIFESTS,
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
    assert [(step.stage_id, step.stage_label) for step in plan.steps] == [
        ("inspect", "Inspect workspace"),
        ("position-mug", "Position the mug"),
        ("position-mug", "Position the mug"),
        ("brew", "Brew coffee"),
        ("verify", "Verify outcome"),
    ]
    assert plan.step_at(0.5).skill == "place"
    assert plan.step_at(1.0).skill == "verify"


def test_offline_planner_builds_typed_drawer_plan() -> None:
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(text="Open the drawer", task="OpenDrawer")
    )

    assert [step.skill for step in plan.steps] == ["locate", "open", "verify"]
    assert plan.goal.task == "OpenDrawer"


def test_offline_planner_builds_drawer_pick_place_plan() -> None:
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(text="Put the straw in the glass", task="DeliverStraw")
    )

    assert [step.skill for step in plan.steps] == [
        "locate",
        "open",
        "pick",
        "place",
        "verify",
    ]
    assert plan.steps[2].stage_id == "retrieve-straw"


def test_skill_plan_rejects_unknown_dependencies_and_skills() -> None:
    goal = EmbodiedGoal(text="Do it", task="PrepareCoffee")

    with pytest.raises(ValueError, match="unsupported skill"):
        SkillPlan(
            goal=goal,
            steps=(SkillStep(step_id="one", skill="shell_command"),),
        ).validate()

    with pytest.raises(ValueError, match="stages must be contiguous"):
        SkillPlan(
            goal=goal,
            steps=(
                SkillStep(
                    step_id="one",
                    skill="locate",
                    stage_id="inspect",
                    start_fraction=0.0,
                    end_fraction=0.3,
                ),
                SkillStep(
                    step_id="two",
                    skill="pick",
                    stage_id="move",
                    start_fraction=0.3,
                    end_fraction=0.7,
                ),
                SkillStep(
                    step_id="three",
                    skill="verify",
                    stage_id="inspect",
                    start_fraction=0.7,
                    end_fraction=1.0,
                ),
            ),
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


@pytest.mark.parametrize(
    "task, expected_stages",
    [
        ("PackIdenticalLunches", 4),
        ("LoadDishwasher", 5),
        ("LoadFridgeByType", 4),
        ("ArrangeDrinkware", 4),
        ("OrganizeCondiments", 5),
        ("StackBowlsCabinet", 4),
        ("RestockPantry", 4),
    ],
)
def test_curated_composite_manifests_have_ordered_subplans(
    task: str,
    expected_stages: int,
) -> None:
    plan = OfflineEmbodiedPlanner().plan(EmbodiedGoal(task=task))
    stage_ids = list(dict.fromkeys(step.stage_id for step in plan.steps))

    assert task in TASK_MANIFESTS
    assert len(stage_ids) == expected_stages
    assert plan.steps[-1].skill == "verify"
    assert len(plan.steps) >= expected_stages


@pytest.mark.parametrize(
    "task, expected_stage_ids, expected_place_target",
    [
        (
            "LoadFridgeByType",
            ["inspect", "vegetable-bowl", "meat-bowl", "verify"],
            "assigned fridge shelf",
        ),
        (
            "ArrangeDrinkware",
            ["inspect", "pitcher", "cup", "verify"],
            "dining counter",
        ),
    ],
)
def test_pick_place_manifests_match_robocasa_task_semantics(
    task: str,
    expected_stage_ids: list[str],
    expected_place_target: str,
) -> None:
    plan = OfflineEmbodiedPlanner().plan(EmbodiedGoal(task=task))

    assert (
        list(dict.fromkeys(step.stage_id for step in plan.steps)) == expected_stage_ids
    )
    assert sum(step.skill == "pick" for step in plan.steps) == 2
    assert sum(step.skill == "place" for step in plan.steps) == 2
    assert all(
        expected_place_target in step.label
        for step in plan.steps
        if step.skill == "place"
    )


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
    assert all(step.stage_id == "execution" for step in plan.steps)

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
