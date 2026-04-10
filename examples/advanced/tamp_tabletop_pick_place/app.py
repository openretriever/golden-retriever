from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RETRIEVER_TAMP_SRC = REPO_ROOT / "packages" / "retriever-tamp" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RETRIEVER_TAMP_SRC) not in sys.path:
    sys.path.insert(0, str(RETRIEVER_TAMP_SRC))

from retriever_tamp.execution.loop import ReplanReason, TAMPController

from bridge import (
    TabletopExecutionAdapter,
    TabletopRefinementProvider,
    TabletopSymbolicModel,
    TabletopTaskPlanner,
    action_from_tamp,
    build_snapshot,
    format_tamp_plan,
)
from domain import (
    DEFAULT_GOAL_ATOMS,
    DEFAULT_INITIAL_STATE,
    action_signature,
    apply_ground_action,
    goals_satisfied,
    pretty_state,
)
from scene import build_demo_scene


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal tabletop TAMP pick-and-place MVP example."
    )
    parser.add_argument(
        "--no-obstacle",
        action="store_true",
        help="Remove the obstacle so the first place candidate succeeds immediately.",
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=3,
        help="Maximum number of blacklist-and-replan attempts after refinement failure.",
    )
    parser.add_argument(
        "--sim",
        choices=("none", "pybullet-direct", "pybullet-gui"),
        default="none",
        help="Execution backend for the tabletop demo. PyBullet modes are optional and require the TAMP environment.",
    )
    parser.add_argument(
        "--gui-sleep",
        type=float,
        default=1.0 / 60.0,
        help="Sleep per PyBullet GUI step to make the animation visible.",
    )
    parser.add_argument(
        "--final-hold-seconds",
        type=float,
        default=0.0,
        help="Keep the GUI viewer alive for a short time after the final state.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scene = build_demo_scene(include_obstacle=not args.no_obstacle)
    symbolic_state = DEFAULT_INITIAL_STATE
    executed_actions = []
    banned_actions: set[str] = set()
    replans = 0

    simulator = None
    if args.sim != "none":
        from pybullet_sim import PyBulletTabletopSimulator, SimConfig

        try:
            simulator = PyBulletTabletopSimulator(
                scene,
                SimConfig(mode=args.sim, gui_sleep_s=args.gui_sleep),
            )
        except RuntimeError as exc:
            print(f"[sim] {exc}")
            return 4

    planner = TabletopTaskPlanner()
    controller = TAMPController(
        symbolic_model=TabletopSymbolicModel(goal_atoms=DEFAULT_GOAL_ATOMS),
        task_planner=planner,
        refinement_provider=TabletopRefinementProvider(scene),
        execution_adapter=TabletopExecutionAdapter(scene, simulator=simulator),
    )

    print("=== TAMP tabletop pick-place MVP ===")
    print(f"Simulator mode: {args.sim}")
    print(scene.summary())
    print(f"Initial symbolic state: {pretty_state(symbolic_state)}")
    print(f"Goal atoms:            {pretty_state(DEFAULT_GOAL_ATOMS)}")
    print()

    try:
        while not goals_satisfied(symbolic_state, DEFAULT_GOAL_ATOMS):
            planner.set_banned_actions(frozenset(banned_actions))
            snapshot = build_snapshot(scene, symbolic_state)
            reason, plan, refinement, feedback = controller.step(snapshot)

            if plan:
                print(f"[task planner] symbolic plan: {format_tamp_plan(plan)}")

            if reason == ReplanReason.NO_PLAN:
                print("[task planner] no plan found for the current symbolic state.")
                return 1

            if reason == ReplanReason.GOAL_REACHED:
                break

            if refinement is not None:
                next_action = action_from_tamp(refinement.action)
                print(f"[tamp] lazily refine next step only: {next_action}")
                tried = ", ".join(refinement.tried_candidates) or "<none>"
                print(f"[motion refiner] tried candidates: {tried}")

                if reason == ReplanReason.REFINEMENT_FAILED:
                    print(f"[motion refiner] refinement failed: {refinement.failure_reason}")
                    banned_actions.add(action_signature(next_action))
                    replans += 1
                    if replans > args.max_replans:
                        print("[tamp] exceeded max replans; stopping.")
                        return 2
                    print(f"[tamp] blacklisted {next_action} and will replan.\n")
                    continue

                if refinement.candidate is not None:
                    print(f"[motion refiner] selected: {refinement.candidate.label}")
                    for primitive in refinement.candidate.primitives:
                        print(f"  - {primitive.name}")

            if reason == ReplanReason.STEP_EXECUTED:
                completed_action = action_from_tamp(
                    feedback.completed_action if feedback and feedback.completed_action else plan[0]
                )
                symbolic_state = apply_ground_action(symbolic_state, completed_action)
                executed_actions.append(completed_action)
                print(f"[executor] symbolic state -> {pretty_state(symbolic_state)}")
                if feedback is not None:
                    scene_summary = feedback.payload.get("scene_summary", scene.compact_summary())
                    print(f"[executor] scene summary  -> {scene_summary}")
                    print(f"[executor] message        -> {feedback.message}")
                print()
                continue

            if reason == ReplanReason.EXECUTION_FAILED:
                print("[executor] execution failed.")
                if feedback is not None:
                    print(f"[executor] message -> {feedback.message}")
                return 3

            if reason == ReplanReason.MONITOR_TRIGGER:
                print("[monitor] execution monitor requested a replan.")
                continue

        print(
            f"Done. Goal satisfied after {len(executed_actions)} actions: "
            f"{' -> '.join(str(action) for action in executed_actions)}"
        )
        return 0
    finally:
        if simulator is not None:
            if args.sim == "pybullet-gui" and args.final_hold_seconds > 0.0:
                simulator.hold(args.final_hold_seconds)
            simulator.close()


if __name__ == "__main__":
    raise SystemExit(main())
