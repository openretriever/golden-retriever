"""Canonical robotics typing standard v1.

There is exactly one class per standard spatial type across the ecosystem:
the runtime's `retriever.types.spatial` payloads. This module re-exports
them (so `retriever_typing.Header` *is* `retriever.types.spatial.Header`).
The classes are `@io`-ready, so they can be used directly as Flow port
payloads.

Registration happens in the runtime (category "spatial", versioned schema
names) — re-registering here would silently override that metadata, so this
module deliberately does not touch the registry. Validators are re-exported
from the runtime as well; keep new validation logic there, not here.
"""

from __future__ import annotations

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
