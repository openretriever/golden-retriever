"""Small runnable demo for robotics typing catalog v1."""

from __future__ import annotations

import sys
from pathlib import Path

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
    Wrench,
    WrenchStamped,
    validate_joint_state,
    validate_pose_stamped,
)
from retriever_typing import get_type


def main() -> None:
    pose_type = get_type("PoseStamped")
    se3_type = get_type("SE3Pose")
    print(f"Registry lookup: PoseStamped={pose_type.__name__} SE3Pose={se3_type.__name__}")

    pose = PoseStamped(
        header=Header(stamp_ns=1_726_000_000_000_000_000, frame_id="map", source="sim"),
        pose=SE3Pose(
            position=Vector3(1.0, 2.0, 0.5),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        ),
    )
    twist = TwistStamped(
        header=Header(stamp_ns=1_726_000_000_000_000_000, frame_id="base_link", source="controller"),
        twist=Twist(
            linear=Vector3(0.2, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.1),
        ),
    )
    wrench = WrenchStamped(
        header=Header(stamp_ns=1_726_000_000_000_000_000, frame_id="wrist", source="ft_sensor"),
        wrench=Wrench(
            force=Vector3(0.0, 0.0, 3.5),
            torque=Vector3(0.1, 0.0, 0.0),
        ),
    )
    joints = JointState(
        names=("j1", "j2", "j3", "j4", "j5", "j6"),
        positions=(0.0, -0.3, 0.8, 0.0, 0.6, 0.0),
        velocities=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        efforts=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    validate_pose_stamped(pose)
    validate_joint_state(joints)

    print("Robotics typing catalog v1 demo")
    print(f"  pose.frame={pose.header.frame_id} pos={pose.pose.position}")
    print(f"  twist.frame={twist.header.frame_id} linear={twist.twist.linear}")
    print(f"  wrench.frame={wrench.header.frame_id} force={wrench.wrench.force}")
    print(f"  joint_count={len(joints.names)} aligned={joints.is_aligned()}")


if __name__ == "__main__":
    main()
