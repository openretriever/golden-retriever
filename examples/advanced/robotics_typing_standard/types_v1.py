"""Robotics typing catalog v1 used by advanced typing demos."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class Header:
    stamp_ns: int
    frame_id: str
    source: str = "unknown"


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def norm(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)

    def is_unit(self, tol: float = 1e-3) -> bool:
        return abs(self.norm() - 1.0) <= tol


@dataclass(frozen=True)
class SE3Pose:
    position: Vector3
    orientation: Quaternion


@dataclass(frozen=True)
class Twist:
    linear: Vector3
    angular: Vector3


@dataclass(frozen=True)
class Wrench:
    force: Vector3
    torque: Vector3


@dataclass(frozen=True)
class JointState:
    names: tuple[str, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    efforts: tuple[float, ...]

    def is_aligned(self) -> bool:
        n = len(self.names)
        return (
            len(self.positions) == n
            and len(self.velocities) == n
            and len(self.efforts) == n
        )


@dataclass(frozen=True)
class PoseStamped:
    header: Header
    pose: SE3Pose


@dataclass(frozen=True)
class TwistStamped:
    header: Header
    twist: Twist


@dataclass(frozen=True)
class WrenchStamped:
    header: Header
    wrench: Wrench


def validate_pose_stamped(msg: PoseStamped) -> None:
    if not msg.header.frame_id:
        raise ValueError("frame_id must be non-empty")
    if msg.header.stamp_ns <= 0:
        raise ValueError("stamp_ns must be > 0")
    if not msg.pose.orientation.is_unit():
        raise ValueError("orientation quaternion must be unit-norm")


def validate_joint_state(msg: JointState) -> None:
    if not msg.is_aligned():
        raise ValueError("joint state arrays must align by length")
