from __future__ import annotations

from golden_retriever.robotics_typing import (
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
)
from golden_retriever.types import get_type


def test_registry_lookup_for_v1_types() -> None:
    expected = {
        "Header": Header,
        "Vector3": Vector3,
        "Quaternion": Quaternion,
        "SE3Pose": SE3Pose,
        "PoseStamped": PoseStamped,
        "Twist": Twist,
        "TwistStamped": TwistStamped,
        "Wrench": Wrench,
        "WrenchStamped": WrenchStamped,
        "JointState": JointState,
    }
    for name, cls in expected.items():
        assert get_type(name) is cls
