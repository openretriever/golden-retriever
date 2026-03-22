from __future__ import annotations

import argparse

from domain import (
    DEFAULT_GOAL_ATOMS,
    DEFAULT_INITIAL_STATE,
    action_signature,
    apply_ground_action,
    goals_satisfied,
    pretty_state,
)
from motion_refiner import refine_action
from scene import build_demo_scene
from task_planner import format_plan, task_plan


def main() -> int:
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
    args = parser.parse_args()

    scene = build_demo_scene(include_obstacle=not args.no_obstacle)
    symbolic_state = DEFAULT_INITIAL_STATE
    executed_actions = []
    banned_actions: set[str] = set()
    replans = 0

    print("=== TAMP tabletop pick-place MVP ===")
    print(scene.summary())
    print(f"Initial symbolic state: {pretty_state(symbolic_state)}")
    print(f"Goal atoms:            {pretty_state(DEFAULT_GOAL_ATOMS)}")
    print()

    while not goals_satisfied(symbolic_state, DEFAULT_GOAL_ATOMS):
        plan = task_plan(
            symbolic_state,
            DEFAULT_GOAL_ATOMS,
            banned_action_signatures=frozenset(banned_actions),
        )
        if not plan:
            print("[task planner] no plan found for the current symbolic state.")
            return 1

        next_action = plan[0]
        print(f"[task planner] symbolic plan: {format_plan(plan)}")
        print(f"[tamp] lazily refine next step only: {next_action}")

        refinement = refine_action(scene, next_action)
        tried = ", ".join(refinement.tried_candidates) or "<none>"
        print(f"[motion refiner] tried candidates: {tried}")

        if not refinement.success:
            print(f"[motion refiner] refinement failed: {refinement.failure_reason}")
            banned_actions.add(action_signature(next_action))
            replans += 1
            if replans > args.max_replans:
                print("[tamp] exceeded max replans; stopping.")
                return 2
            print(f"[tamp] blacklisted {next_action} and will replan.\n")
            continue

        assert refinement.segment is not None
        print(f"[motion refiner] selected: {refinement.segment.candidate_label}")
        for command in refinement.segment.commands:
            print(f"  - {command}")

        if next_action.name == "Pick":
            scene.commit_pick(next_action.args[0])
        elif next_action.name == "Place":
            scene.commit_place(next_action.args[0], refinement.segment.target_pose)

        symbolic_state = apply_ground_action(symbolic_state, next_action)
        executed_actions.append(next_action)

        print(f"[executor] symbolic state -> {pretty_state(symbolic_state)}")
        print(f"[executor] scene summary  -> {scene.compact_summary()}")
        print()

    print(
        f"Done. Goal satisfied after {len(executed_actions)} actions: "
        f"{format_plan(executed_actions)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
