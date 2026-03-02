from __future__ import annotations

from golden_retriever.robotics_typing import PoseStamped as PublicPoseStamped
from golden_retriever.robotics_typing import SE3Pose as PublicSE3Pose
from golden_retriever.robotics_typing.v1 import PoseStamped as PinnedPoseStamped
from golden_retriever.robotics_typing.v1 import SE3Pose as PinnedSE3Pose
from golden_retriever.types import get_type


def test_public_package_surface_matches_pinned_v1() -> None:
    assert PublicPoseStamped is PinnedPoseStamped
    assert PublicSE3Pose is PinnedSE3Pose


def test_public_surface_matches_registry_lookup() -> None:
    assert get_type("PoseStamped") is PublicPoseStamped
    assert get_type("SE3Pose") is PublicSE3Pose
