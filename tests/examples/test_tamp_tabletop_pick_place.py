from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "advanced" / "tamp_tabletop_pick_place"
RETRIEVER_TAMP_SRC = REPO_ROOT / "packages" / "retriever-tamp" / "src"
APP_PATH = REPO_ROOT / "examples" / "advanced" / "tamp_tabletop_pick_place" / "app.py"

if str(RETRIEVER_TAMP_SRC) not in sys.path:
    sys.path.insert(0, str(RETRIEVER_TAMP_SRC))
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from retriever_tamp.core.types import GoalSpec, GroundAction, GroundAtom  # noqa: E402
from retriever_tamp.refinement.base import RefinementRequest  # noqa: E402
from retriever_tamp.symbolic.base import TaskPlanningProblem  # noqa: E402

from bridge import (  # noqa: E402
    TabletopExecutionAdapter,
    TabletopRefinementProvider,
    TabletopSymbolicModel,
    TabletopTaskPlanner,
    build_snapshot,
)
from domain import (  # noqa: E402
    DEFAULT_GOAL_ATOMS,
    DEFAULT_INITIAL_STATE,
    action_signature,
    apply_ground_action,
    atom,
    goals_satisfied,
)
from scene import build_demo_scene  # noqa: E402


def _run_demo(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"packages/retriever-tamp/src:."
    return subprocess.run(
        [sys.executable, str(APP_PATH), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_tabletop_tamp_demo_no_sim_runs() -> None:
    result = _run_demo("--no-obstacle")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Done. Goal satisfied" in result.stdout


def test_tabletop_tamp_demo_default_obstacle_path_replans_successfully() -> None:
    result = _run_demo()
    assert result.returncode == 0, result.stderr or result.stdout
    assert "place-left-entry@goal_region, place-top-entry@goal_region" in result.stdout
    assert "selected: place-top-entry@goal_region" in result.stdout


def test_tabletop_tamp_demo_pybullet_direct_runs_if_installed() -> None:
    pytest.importorskip("pybullet")
    result = _run_demo("--sim", "pybullet-direct", "--no-obstacle")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Simulator mode: pybullet-direct" in result.stdout


def test_tamp_domain_uses_shared_symbolic_core_types() -> None:
    pick = GroundAction("Pick", ("red_block", "start_region"))
    place = GroundAction("Place", ("red_block", "goal_region"))

    assert isinstance(atom("HandEmpty"), GroundAtom)
    assert isinstance(DEFAULT_INITIAL_STATE, frozenset)
    assert all(isinstance(item, GroundAtom) for item in DEFAULT_INITIAL_STATE)
    assert all(isinstance(item, GroundAtom) for item in DEFAULT_GOAL_ATOMS)

    state = apply_ground_action(DEFAULT_INITIAL_STATE, pick)
    assert isinstance(state, frozenset)
    assert all(isinstance(item, GroundAtom) for item in state)
    assert not goals_satisfied(state, DEFAULT_GOAL_ATOMS)

    state = apply_ground_action(state, place)
    assert goals_satisfied(state, DEFAULT_GOAL_ATOMS)


def test_tamp_bridge_uses_shared_symbolic_problem_contracts() -> None:
    scene = build_demo_scene(include_obstacle=False)
    snapshot = build_snapshot(scene, DEFAULT_INITIAL_STATE)
    model = TabletopSymbolicModel(goal_atoms=DEFAULT_GOAL_ATOMS)
    planner = TabletopTaskPlanner()

    abstract_state = model.abstract(snapshot)
    assert abstract_state == DEFAULT_INITIAL_STATE

    goal = model.goal(snapshot)
    assert isinstance(goal, GoalSpec)
    assert goal.required_atoms == DEFAULT_GOAL_ATOMS

    operators = tuple(model.operators(snapshot))
    assert operators
    assert operators[0].preconditions
    assert all(isinstance(atom_, GroundAtom) for atom_ in operators[0].preconditions)

    problem = TaskPlanningProblem(
        initial_state=abstract_state,
        goal=goal,
        operators=operators,
    )
    plan = tuple(planner.plan(problem))
    assert plan
    assert all(isinstance(action, GroundAction) for action in plan)


def test_tamp_execution_adapter_updates_scene_through_shared_actions() -> None:
    scene = build_demo_scene(include_obstacle=False)
    provider = TabletopRefinementProvider(scene)
    adapter = TabletopExecutionAdapter(scene)
    model = TabletopSymbolicModel(goal_atoms=DEFAULT_GOAL_ATOMS)
    planner = TabletopTaskPlanner()

    snapshot = build_snapshot(scene, DEFAULT_INITIAL_STATE)
    problem = TaskPlanningProblem(
        initial_state=model.abstract(snapshot),
        goal=model.goal(snapshot),
        operators=tuple(model.operators(snapshot)),
    )
    plan = tuple(planner.plan(problem))
    assert action_signature(plan[0]) == "Pick(red_block, start_region)"

    pick_refinement = provider.refine(
        RefinementRequest(action=plan[0], snapshot=snapshot)
    )
    pick_feedback = adapter.execute(pick_refinement)
    assert pick_feedback.success
    assert scene.held_object == "red_block"

    state = apply_ground_action(DEFAULT_INITIAL_STATE, plan[0])
    next_snapshot = build_snapshot(scene, state)
    next_problem = TaskPlanningProblem(
        initial_state=model.abstract(next_snapshot),
        goal=model.goal(next_snapshot),
        operators=tuple(model.operators(next_snapshot)),
    )
    place_action = tuple(planner.plan(next_problem))[0]
    place_refinement = provider.refine(
        RefinementRequest(action=place_action, snapshot=next_snapshot)
    )
    place_feedback = adapter.execute(place_refinement)

    assert place_feedback.success
    assert scene.held_object is None
    final_snapshot = build_snapshot(scene, apply_ground_action(state, place_action))
    assert model.goal(final_snapshot).is_satisfied_by(final_snapshot.symbolic_state)
