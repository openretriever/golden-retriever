from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
    """Spatial standard types carry the runtime's canonical registration —
    the unified registry means Golden sees core's metadata, not a parallel copy."""
    registry = get_registered_types()

    pose_info = registry["PoseStamped"]
    assert pose_info.type_class is PoseStamped
    assert pose_info.category == "spatial"
    assert "pose" in pose_info.tags
    assert pose_info.schema_name == "spatial/PoseStamped"

    joint_info = registry["JointState"]
    assert joint_info.type_class is JointState
    assert joint_info.category == "spatial"
    assert "joint" in joint_info.tags


def test_v1_types_are_the_runtime_standard_types() -> None:
    """retriever_typing re-exports the runtime's canonical spatial standard:
    one class per standard type across the ecosystem, not a parallel copy."""
    import retriever.types.spatial as spatial

    for cls in (Header, Vector3, Quaternion, SE3Pose, PoseStamped,
                Twist, TwistStamped, Wrench, WrenchStamped, JointState):
        assert cls is getattr(spatial, cls.__name__)


def test_applied_type_registry_surfaces_bootstrap_in_fresh_process() -> None:
    script = """
from retriever_typing.registry import get_type_info, list_types
info = get_type_info("WorldState")
assert info.type_class.__name__ == "WorldState"
assert "WorldState" in list_types()
"""
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    pythonpath = str(repo_root / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
