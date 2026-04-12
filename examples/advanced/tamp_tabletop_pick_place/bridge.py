from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from retriever_tamp.core.types import GoalSpec, GroundAction, SymbolicState, WorldSnapshot
from retriever_tamp.execution.loop import ExecutionAdapter, ExecutionFeedback
from retriever_tamp.refinement.base import (
    ExecutionPrimitive,
    RefinementCandidate,
    RefinementProvider,
    RefinementRequest,
    RefinementResult,
)
from retriever_tamp.symbolic.base import OperatorSchema, SymbolicModel, TaskPlanner, TaskPlanningProblem

from domain import DEFAULT_GOAL_ATOMS, OPERATORS, action_signature
from motion_refiner import MotionSegment, refine_action
from scene import Pose2D, TabletopScene


def format_tamp_plan(plan: Sequence[GroundAction]) -> str:
    if not plan:
        return "<no plan>"
    return " -> ".join(str(action) for action in plan)


def build_snapshot(scene: TabletopScene, symbolic_state: SymbolicState) -> WorldSnapshot:
    region_membership: dict[str, str | None] = {}
    for region_name in (scene.start_region.name, scene.goal_region.name):
        region_membership[region_name] = None
        if scene.region(region_name).contains(scene.block_pose):
            region_membership[region_name] = scene.block_name

    objects = {
        scene.block_name: {
            "pose_xy": (scene.block_pose.x, scene.block_pose.y),
            "held": scene.held_object == scene.block_name,
        },
        scene.start_region.name: {
            "center_xy": (scene.start_region.center.x, scene.start_region.center.y),
            "contains": region_membership[scene.start_region.name],
        },
        scene.goal_region.name: {
            "center_xy": (scene.goal_region.center.x, scene.goal_region.center.y),
            "contains": region_membership[scene.goal_region.name],
        },
    }
    if scene.obstacle is not None:
        objects[scene.obstacle.name] = {
            "center_xy": (scene.obstacle.center.x, scene.obstacle.center.y),
            "radius": scene.obstacle.radius,
        }

    return WorldSnapshot(
        raw_observation=scene,
        symbolic_state=symbolic_state,
        objects=objects,
        metadata={"held_object": scene.held_object, "scene_summary": scene.compact_summary()},
    )


def _operator_to_schema(operator) -> OperatorSchema:
    return OperatorSchema(
        name=operator.name,
        parameters=operator.parameters,
        preconditions=operator.preconditions,
        add_effects=operator.add_effects,
        delete_effects=operator.delete_effects,
    )


@dataclass
class TabletopSymbolicModel(SymbolicModel):
    goal_atoms: SymbolicState = DEFAULT_GOAL_ATOMS

    def abstract(self, snapshot: WorldSnapshot) -> SymbolicState:
        return snapshot.symbolic_state

    def operators(self, snapshot: WorldSnapshot):
        del snapshot
        return tuple(_operator_to_schema(operator) for operator in OPERATORS)

    def goal(self, snapshot: WorldSnapshot) -> GoalSpec:
        del snapshot
        return GoalSpec(required_atoms=self.goal_atoms)


@dataclass
class TabletopTaskPlanner(TaskPlanner):
    banned_action_signatures: frozenset[str] = frozenset()

    def set_banned_actions(self, banned: frozenset[str]) -> None:
        self.banned_action_signatures = banned

    def plan(self, problem: TaskPlanningProblem):
        from task_planner import task_plan

        return tuple(
            task_plan(
                initial_state=problem.initial_state,
                goal_atoms=problem.goal.required_atoms,
                banned_action_signatures=self.banned_action_signatures,
            )
        )


def _make_candidate(segment: MotionSegment) -> RefinementCandidate:
    primitives = tuple(
        ExecutionPrimitive(name=command, parameters={})
        for command in segment.commands
    )
    return RefinementCandidate(
        label=segment.candidate_label,
        primitives=primitives,
        metadata={"segment": segment},
    )


@dataclass
class TabletopRefinementProvider(RefinementProvider):
    scene: TabletopScene

    def refine(self, request: RefinementRequest) -> RefinementResult:
        result = refine_action(self.scene, request.action)
        if not result.success or result.segment is None:
            return RefinementResult(
                action=request.action,
                success=False,
                tried_candidates=result.tried_candidates,
                failure_reason=result.failure_reason,
            )
        return RefinementResult(
            action=request.action,
            success=True,
            tried_candidates=result.tried_candidates,
            candidate=_make_candidate(result.segment),
        )


class TabletopExecutionAdapter(ExecutionAdapter):
    def __init__(self, scene: TabletopScene, simulator=None) -> None:
        self._scene = scene
        self._simulator = simulator

    def execute(self, refinement: RefinementResult) -> ExecutionFeedback:
        segment = None
        if refinement.candidate is not None:
            segment = refinement.candidate.metadata.get("segment")
        if not refinement.success or segment is None:
            return ExecutionFeedback(
                success=False,
                completed_action=refinement.action,
                message="Execution skipped because refinement produced no executable segment.",
            )

        if self._simulator is not None:
            try:
                self._simulator.execute(refinement.action, segment, self._scene)
            except Exception as exc:
                return ExecutionFeedback(
                    success=False,
                    completed_action=refinement.action,
                    message=f"Simulator execution failed: {exc}",
                )

        if refinement.action.name == "Pick":
            self._scene.commit_pick(refinement.action.args[0])
        elif refinement.action.name == "Place":
            self._scene.commit_place(refinement.action.args[0], segment.target_pose)
        else:
            return ExecutionFeedback(
                success=False,
                completed_action=refinement.action,
                message=f"Unsupported action type: {refinement.action.name}",
            )

        return ExecutionFeedback(
            success=True,
            completed_action=refinement.action,
            message=f"Executed {action_signature(refinement.action)}",
            payload={"scene_summary": self._scene.compact_summary()},
        )


def pose_to_xy(pose: Pose2D) -> tuple[float, float]:
    return (pose.x, pose.y)
