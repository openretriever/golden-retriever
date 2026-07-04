"""Arrow round-trip fidelity tests for retriever_typing conversions.

Every serializable payload should survive convert_to_arrow → convert_from_arrow
with values, dtypes, and shapes intact.
"""

from __future__ import annotations

import numpy as np
import pytest

from retriever_typing import (
    Action,
    Command,
    Header,
    JointState,
    PoseStamped,
    Quaternion,
    SE3Pose,
    Status,
    Vector3,
    convert_from_arrow,
    convert_to_arrow,
)


def roundtrip(obj):
    return convert_from_arrow(convert_to_arrow(obj))


def assert_array_identical(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)






def test_action_command_status_roundtrip() -> None:
    action = Action(type="grasp", parameters={"object": "cup", "force": 0.4}, timestamp=2.0, priority=3)
    result_action = roundtrip(action)
    assert result_action == action

    command = Command(action=action, robot_id="r1", expected_duration=1.5, timeout=4.0)
    result_command = roundtrip(command)
    assert result_command == command

    status = Status(state="running", message="in progress", progress=42.0, timestamp=3.0)
    result_status = roundtrip(status)
    assert result_status == status


def test_robotics_v1_pose_stamped_roundtrip() -> None:
    msg = PoseStamped(
        header=Header(stamp_ns=123, frame_id="map", source="test"),
        pose=SE3Pose(
            position=Vector3(x=1.0, y=2.0, z=3.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )
    assert roundtrip(msg) == msg


def test_robotics_v1_joint_state_roundtrip() -> None:
    msg = JointState(
        names=("j1", "j2"),
        positions=(0.1, 0.2),
        velocities=(1.0, -1.0),
        efforts=(0.0, 0.5),
    )
    assert roundtrip(msg) == msg


def test_unregistered_payload_falls_back_to_json() -> None:
    payload = {"plain": "dict", "n": 3}
    assert roundtrip(payload) == payload


def test_perception_image2d_roundtrip() -> None:
    from retriever.types.perception import Image2D
    from retriever.types.spatial import Header as SpatialHeader

    data = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    header = SpatialHeader(stamp_ns=123, frame_id="cam", source="test")

    result = roundtrip(Image2D(data=data, encoding="rgb8", header=header, frame_index=7))
    assert isinstance(result, Image2D)
    assert_array_identical(result.data, data)
    assert result.encoding == "rgb8"
    assert result.header == header
    assert result.frame_index == 7

    headerless = roundtrip(Image2D(data=data))
    assert headerless.header is None


def test_perception_detection_batch_roundtrip() -> None:
    from retriever.types.perception import BBox2D, Detection2D, DetectionBatch

    det = Detection2D(
        label="cup",
        bbox=BBox2D(x=1.0, y=2.0, width=3.0, height=4.0),
        confidence=0.9,
        track_id="t1",
    )
    batch = DetectionBatch(detections=(det,), header=None, frame_index=3)

    result = roundtrip(batch)
    assert isinstance(result, DetectionBatch)
    assert result.frame_index == 3
    assert len(result.detections) == 1
    assert result.detections[0] == det

    assert roundtrip(det) == det
    assert roundtrip(det.bbox) == det.bbox


def test_perception_mask_and_point_target_roundtrip() -> None:
    from retriever.types.perception import PointTarget2D, SegmentationMask2D

    mask = np.zeros((4, 5), dtype=np.int32)
    mask[1, 2] = 7
    seg = SegmentationMask2D(mask=mask, label_map={0: "background", 7: "cup"})
    result = roundtrip(seg)
    assert isinstance(result, SegmentationMask2D)
    assert_array_identical(result.mask, mask)
    assert result.label_map == {0: "background", 7: "cup"}

    target = PointTarget2D(label="cup", x_norm=0.5, y_norm=0.25, confidence=0.8)
    assert roundtrip(target) == target
