"""Representative stamped-boundary walkthrough for robotics typing.

This example keeps the runtime lightweight but models a real boundary chain:
camera perception -> frame normalization -> control command -> serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from retriever_typing import (
    Header,
    JointState,
    PoseStamped,
    Quaternion,
    SE3Pose,
    Twist,
    TwistStamped,
    Vector3,
    validate_joint_state,
    validate_pose_stamped,
)
from golden_retriever.types import convert_from_arrow, convert_to_arrow


@dataclass(frozen=True)
class ApproachDecision:
    target: PoseStamped
    command: TwistStamped


def detect_target_from_camera() -> PoseStamped:
    target = PoseStamped(
        header=Header(
            stamp_ns=1_726_100_000_000_000_000,
            frame_id="camera_color_optical_frame",
            source="rgb_detector",
        ),
        pose=SE3Pose(
            position=Vector3(0.42, -0.08, 0.63),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        ),
    )
    validate_pose_stamped(target)
    return target


def normalize_target_to_base(camera_target: PoseStamped) -> PoseStamped:
    validate_pose_stamped(camera_target)

    # Mock frame transform step. The important contract is that the frame/source
    # transition is explicit in the typed payload rather than implied.
    base_target = PoseStamped(
        header=Header(
            stamp_ns=camera_target.header.stamp_ns + 500_000,
            frame_id="base_link",
            source="frame_projection",
        ),
        pose=SE3Pose(
            position=Vector3(
                x=camera_target.pose.position.x - 0.12,
                y=camera_target.pose.position.y,
                z=camera_target.pose.position.z - 0.18,
            ),
            orientation=camera_target.pose.orientation,
        ),
    )
    validate_pose_stamped(base_target)
    return base_target


def plan_cartesian_approach(target: PoseStamped, joints: JointState) -> ApproachDecision:
    validate_pose_stamped(target)
    validate_joint_state(joints)

    mean_joint = sum(joints.positions) / len(joints.positions)
    linear_x = max(0.0, target.pose.position.x - 0.25) + mean_joint * 0.05
    linear_y = target.pose.position.y * 0.4

    command = TwistStamped(
        header=Header(
            stamp_ns=target.header.stamp_ns + 1_000_000,
            frame_id="base_link",
            source="approach_controller",
        ),
        twist=Twist(
            linear=Vector3(linear_x, linear_y, 0.0),
            angular=Vector3(0.0, 0.0, 0.15),
        ),
    )
    return ApproachDecision(target=target, command=command)


def main() -> None:
    joints = JointState(
        names=("shoulder", "elbow", "wrist"),
        positions=(0.10, -0.25, 0.35),
        velocities=(0.0, 0.0, 0.0),
        efforts=(0.0, 0.0, 0.0),
    )
    validate_joint_state(joints)

    camera_target = detect_target_from_camera()
    base_target = normalize_target_to_base(camera_target)
    decision = plan_cartesian_approach(base_target, joints)

    serialized = convert_to_arrow(decision.command)
    recovered = convert_from_arrow(serialized)

    print("Perception to control boundary demo")
    print(
        "  camera_target:",
        camera_target.header.frame_id,
        camera_target.header.source,
        camera_target.pose.position,
    )
    print(
        "  base_target:",
        decision.target.header.frame_id,
        decision.target.header.source,
        decision.target.pose.position,
    )
    print(
        "  command:",
        decision.command.header.frame_id,
        decision.command.header.source,
        decision.command.twist.linear,
    )
    print(
        "  serialized_roundtrip:",
        type(recovered).__name__,
        recovered.header.frame_id,
        recovered.twist.linear,
    )


if __name__ == "__main__":
    main()
