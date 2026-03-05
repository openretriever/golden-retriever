"""Compositional typing demo with strict collision behavior.

This script mirrors the desired runtime contract for:
- Flow[(A, B), C]
- unique unqualified access
- ambiguous unqualified access raising
- qualified access for collisions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from retriever_typing import (
    Header,
    JointState,
    PoseStamped,
    Quaternion,
    SE3Pose,
    Vector3,
    validate_joint_state,
    validate_pose_stamped,
)


class AmbiguousFieldError(RuntimeError):
    pass


class FieldNotFoundError(RuntimeError):
    pass


@dataclass
class MotionCommand:
    target_x: float
    gripper_open: bool


class _AliasView:
    def __init__(self, data: Any):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if hasattr(self._data, name):
            return getattr(self._data, name)
        raise FieldNotFoundError(f"field not found on alias view: {name}")


class CompositeIOView:
    """Local demo implementation of IO collision routing."""

    def __init__(self, payloads: dict[str, Any]):
        self._payloads = payloads
        self._field_to_aliases: dict[str, list[str]] = {}
        for alias, obj in payloads.items():
            for field in vars(obj).keys():
                self._field_to_aliases.setdefault(field, []).append(alias)

    def __getattr__(self, name: str) -> Any:
        if name in self._payloads:
            return _AliasView(self._payloads[name])

        aliases = self._field_to_aliases.get(name)
        if not aliases:
            raise FieldNotFoundError(f"field not found: {name}")
        if len(aliases) > 1:
            raise AmbiguousFieldError(
                f"ambiguous field '{name}' present in {aliases}; use qualified access"
            )
        alias = aliases[0]
        return getattr(self._payloads[alias], name)

    def _get_signal(self, key: str) -> Any:
        if "." in key:
            alias, field = key.split(".", 1)
            if alias not in self._payloads:
                raise FieldNotFoundError(f"alias not found: {alias}")
            obj = self._payloads[alias]
            if not hasattr(obj, field):
                raise FieldNotFoundError(f"field not found: {key}")
            return getattr(obj, field)
        return getattr(self, key)


def planner_step(inp: CompositeIOView) -> MotionCommand:
    # Unique unqualified field access.
    joint_positions = inp.positions
    mean_position = sum(joint_positions) / max(1, len(joint_positions))

    # Collision-safe qualified field access.
    goal_x = inp.pose.pose.position.x
    base_x = inp.base.pose.position.x

    target_x = goal_x - base_x + mean_position
    return MotionCommand(target_x=target_x, gripper_open=True)


def main() -> None:
    goal_pose = PoseStamped(
        header=Header(stamp_ns=100, frame_id="map", source="planner"),
        pose=SE3Pose(
            position=Vector3(1.2, 0.0, 0.4),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        ),
    )
    base_pose = PoseStamped(
        header=Header(stamp_ns=100, frame_id="base_link", source="state_estimator"),
        pose=SE3Pose(
            position=Vector3(0.3, 0.0, 0.4),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        ),
    )
    joint_state = JointState(
        names=("j1", "j2", "j3"),
        positions=(0.1, -0.1, 0.2),
        velocities=(0.0, 0.0, 0.0),
        efforts=(0.0, 0.0, 0.0),
    )

    validate_pose_stamped(goal_pose)
    validate_pose_stamped(base_pose)
    validate_joint_state(joint_state)

    # Flow[(PoseStamped, PoseStamped, JointState), MotionCommand]-style composite view.
    view = CompositeIOView({"pose": goal_pose, "base": base_pose, "joint": joint_state})

    print("Unique field access:")
    print(f"  names = {view.names}")

    print("Ambiguous field access (expected error):")
    try:
        _ = view.header
    except AmbiguousFieldError as err:
        print(f"  {err}")

    print("Qualified access:")
    print(f"  pose.header.frame_id = {view.pose.header.frame_id}")
    print(f"  base.header.frame_id = {view.base.header.frame_id}")

    cmd = planner_step(view)
    print("Planner output:")
    print(f"  MotionCommand(target_x={cmd.target_x:.3f}, gripper_open={cmd.gripper_open})")


if __name__ == "__main__":
    main()
