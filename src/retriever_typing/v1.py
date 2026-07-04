"""Canonical robotics typing standard v1.

There is exactly one class per standard spatial type across the ecosystem:
the runtime's `retriever.types.spatial` payloads. This module re-exports
them (so `retriever_typing.Header` *is* `retriever.types.spatial.Header`)
and registers them in the `retriever_typing` registry with robotics
metadata. The classes are `@io`-ready, so they can be used directly as Flow
port payloads.

Validators are re-exported from the runtime as well; keep new validation
logic there, not here.
"""

from __future__ import annotations

from typing import Final

from retriever.types.spatial import (
    Header,
    JointState,
    PoseStamped,
    Quaternion,
    SE3Pose,
    Twist,
    TwistStamped,
    Vector3,
    Wrench,
    WrenchStamped,
    validate_header,
    validate_joint_state,
    validate_pose_stamped,
    validate_quaternion,
)

from .registry import register_type

_ROBOTICS_CATEGORY: Final[str] = "robotics"

_V1_TYPES: Final[tuple[tuple[type, str, list[str]], ...]] = (
    (Header, "Header for stamped robotics payloads", ["robotics", "v1", "header", "metadata"]),
    (Vector3, "3D vector payload", ["robotics", "v1", "geometry", "vector"]),
    (Quaternion, "Quaternion rotation payload", ["robotics", "v1", "geometry", "quaternion"]),
    (SE3Pose, "SE(3) pose payload", ["robotics", "v1", "geometry", "pose"]),
    (Twist, "Spatial velocity payload", ["robotics", "v1", "motion", "twist"]),
    (Wrench, "Force and torque payload", ["robotics", "v1", "force", "wrench"]),
    (JointState, "Joint state payload", ["robotics", "v1", "joint", "state"]),
    (PoseStamped, "Timestamped pose payload", ["robotics", "v1", "pose", "stamped"]),
    (TwistStamped, "Timestamped twist payload", ["robotics", "v1", "twist", "stamped"]),
    (WrenchStamped, "Timestamped wrench payload", ["robotics", "v1", "wrench", "stamped"]),
)

for _cls, _description, _tags in _V1_TYPES:
    register_type(description=_description, category=_ROBOTICS_CATEGORY, tags=_tags)(_cls)


__all__ = [
    "Header",
    "Vector3",
    "Quaternion",
    "SE3Pose",
    "PoseStamped",
    "Twist",
    "TwistStamped",
    "Wrench",
    "WrenchStamped",
    "JointState",
    "validate_header",
    "validate_quaternion",
    "validate_pose_stamped",
    "validate_joint_state",
]
