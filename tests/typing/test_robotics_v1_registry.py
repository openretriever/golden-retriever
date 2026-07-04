from __future__ import annotations

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
)
from retriever_typing import get_registered_types, get_type


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


def test_registry_metadata_for_representative_spatial_types() -> None:
    registry = get_registered_types()

    pose_info = registry["PoseStamped"]
    assert pose_info.type_class is PoseStamped
    assert pose_info.category == "robotics"
    assert "robotics" in pose_info.tags
    assert "pose" in pose_info.tags

    joint_info = registry["JointState"]
    assert joint_info.type_class is JointState
    assert joint_info.category == "robotics"
    assert "joint" in joint_info.tags


def test_v1_types_are_the_runtime_standard_types() -> None:
    """retriever_typing re-exports the runtime's canonical spatial standard:
    one class per standard type across the ecosystem, not a parallel copy."""
    import retriever.types.spatial as spatial

    for cls in (Header, Vector3, Quaternion, SE3Pose, PoseStamped,
                Twist, TwistStamped, Wrench, WrenchStamped, JointState):
        assert cls is getattr(spatial, cls.__name__)
