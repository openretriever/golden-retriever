from __future__ import annotations

from retriever_typing import Header as PublicHeader
from retriever_typing import JointState as PublicJointState
from retriever_typing import PoseStamped as PublicPoseStamped
from retriever_typing import SE3Pose as PublicSE3Pose
from retriever_typing.data import DataSpec as PublicDataSpec
from retriever_typing.data import Event as PublicEvent
from retriever_typing.data import StreamSpec as PublicStreamSpec
from retriever_typing.data.v1 import DataSpec as PinnedDataSpec
from retriever_typing.data.v1 import Event as PinnedEvent
from retriever_typing.data.v1 import StreamSpec as PinnedStreamSpec
from retriever_typing.v1 import Header as PinnedHeader
from retriever_typing.v1 import JointState as PinnedJointState
from retriever_typing.v1 import PoseStamped as PinnedPoseStamped
from retriever_typing.v1 import SE3Pose as PinnedSE3Pose
from retriever_typing import get_arrow_converter
from retriever_typing import get_type
from retriever_typing import get_type_info
from retriever_typing import get_type_name
from retriever_typing import is_registered_type
from retriever_typing import list_types


def test_public_package_surface_matches_pinned_v1() -> None:
    assert PublicPoseStamped is PinnedPoseStamped
    assert PublicSE3Pose is PinnedSE3Pose


def test_public_root_surface_covers_representative_spatial_types() -> None:
    assert PublicHeader is PinnedHeader
    assert PublicJointState is PinnedJointState


def test_public_data_surface_matches_pinned_v1() -> None:
    assert PublicEvent is PinnedEvent
    assert PublicDataSpec is PinnedDataSpec
    assert PublicStreamSpec is PinnedStreamSpec


def test_public_surface_matches_registry_lookup() -> None:
    assert get_type("PoseStamped") is PublicPoseStamped
    assert get_type("SE3Pose") is PublicSE3Pose
    assert get_type("Header") is PublicHeader
    assert get_type("JointState") is PublicJointState


def test_public_root_exports_registry_metadata_helpers() -> None:
    from retriever_typing import WorldState

    assert get_type_info("WorldState").type_class is WorldState
    assert list_types()["WorldState"].type_class is WorldState


def test_public_root_preserves_legacy_registry_helpers() -> None:
    assert is_registered_type(PublicPoseStamped)
    assert get_type_name(PublicPoseStamped) == "PoseStamped"
    assert get_arrow_converter(PublicPoseStamped) is None
