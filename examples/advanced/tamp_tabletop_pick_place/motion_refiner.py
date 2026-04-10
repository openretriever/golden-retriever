from __future__ import annotations

from dataclasses import dataclass

from domain import GroundAction
from scene import MotionCandidate, Pose2D, TabletopScene


@dataclass(frozen=True)
class MotionSegment:
    candidate_label: str
    target_pose: Pose2D
    approach_pose: Pose2D
    retreat_pose: Pose2D
    commands: tuple[str, ...]


@dataclass(frozen=True)
class RefinementResult:
    action: GroundAction
    success: bool
    tried_candidates: tuple[str, ...]
    segment: MotionSegment | None = None
    failure_reason: str = ""


def refine_action(scene: TabletopScene, action: GroundAction) -> RefinementResult:
    object_name = action.args[0]

    if action.name == "Pick":
        candidates = scene.pick_candidates(object_name)
    elif action.name == "Place":
        region_name = action.args[1]
        candidates = scene.place_candidates(region_name)
    else:
        return RefinementResult(
            action=action,
            success=False,
            tried_candidates=(),
            failure_reason=f"Unsupported action type: {action.name}",
        )

    tried: list[str] = []
    for candidate in candidates:
        tried.append(candidate.label)
        if not scene.candidate_feasible(candidate):
            continue
        return RefinementResult(
            action=action,
            success=True,
            tried_candidates=tuple(tried),
            segment=_make_segment(action, candidate),
        )

    return RefinementResult(
        action=action,
        success=False,
        tried_candidates=tuple(tried),
        failure_reason="No feasible motion candidate for the next symbolic step.",
    )


def _make_segment(action: GroundAction, candidate: MotionCandidate) -> MotionSegment:
    if action.name == "Pick":
        commands = (
            f"MoveTCP(approach=({candidate.approach_pose.x:.2f}, {candidate.approach_pose.y:.2f}))",
            f"MoveTCP(grasp=({candidate.target_pose.x:.2f}, {candidate.target_pose.y:.2f}))",
            "CloseGripper()",
            f"Lift(retreat=({candidate.retreat_pose.x:.2f}, {candidate.retreat_pose.y:.2f}))",
        )
    else:
        commands = (
            f"MoveTCP(approach=({candidate.approach_pose.x:.2f}, {candidate.approach_pose.y:.2f}))",
            f"MoveTCP(place=({candidate.target_pose.x:.2f}, {candidate.target_pose.y:.2f}))",
            "OpenGripper()",
            f"RetreatTCP(to=({candidate.retreat_pose.x:.2f}, {candidate.retreat_pose.y:.2f}))",
        )

    return MotionSegment(
        candidate_label=candidate.label,
        target_pose=candidate.target_pose,
        approach_pose=candidate.approach_pose,
        retreat_pose=candidate.retreat_pose,
        commands=commands,
    )
