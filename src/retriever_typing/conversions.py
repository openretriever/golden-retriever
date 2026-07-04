"""PyArrow conversion utilities for Retriever types."""

from __future__ import annotations

import base64
import json
import pickle
from typing import Any, Callable, Dict, Optional, Type

import numpy as np
import pyarrow as pa

from .v1 import (
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
from .core_types import (
    Action,
    Command,
    ExecutionTimer,
    Status,
    Timestamp,
)
from retriever.types.perception import (
    BBox2D,
    Detection2D,
    DetectionBatch,
    Image2D,
    PointTarget2D,
    SegmentationMask2D,
)

_arrow_converters: Dict[Type, Callable[[Any], pa.Array]] = {}
_arrow_deserializers: Dict[Type, Callable[[pa.Array], Any]] = {}


def register_conversion(
    type_class: Type,
    to_arrow: Callable[[Any], pa.Array],
    from_arrow: Callable[[pa.Array], Any],
) -> None:
    """Register custom PyArrow conversion for a type."""
    _arrow_converters[type_class] = to_arrow
    _arrow_deserializers[type_class] = from_arrow


def _encode_ndarray(arr: np.ndarray) -> dict[str, Any]:
    return {
        "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def _decode_ndarray(payload: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(payload["b64"].encode("ascii"))
    return np.frombuffer(raw, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])


def _pack(payload: dict[str, Any]) -> pa.Array:
    return pa.array([json.dumps(payload)])


def _header_dict(header: Optional[Header]) -> Optional[dict[str, Any]]:
    if header is None:
        return None
    return {"stamp_ns": header.stamp_ns, "frame_id": header.frame_id, "source": header.source}


def _header_from(data: Optional[dict[str, Any]]) -> Optional[Header]:
    if data is None:
        return None
    return Header(
        stamp_ns=data["stamp_ns"],
        frame_id=data["frame_id"],
        source=data.get("source", "unknown"),
    )


def _detection2d_dict(det: "Detection2D") -> dict[str, Any]:
    return {
        "label": det.label,
        "bbox": {"x": det.bbox.x, "y": det.bbox.y, "width": det.bbox.width, "height": det.bbox.height},
        "confidence": det.confidence,
        "class_id": det.class_id,
        "track_id": det.track_id,
        "centroid_x": det.centroid_x,
        "centroid_y": det.centroid_y,
        "pixel_count": det.pixel_count,
    }


def _detection2d_from(data: dict[str, Any]) -> "Detection2D":
    bbox = data["bbox"]
    return Detection2D(
        label=data["label"],
        bbox=BBox2D(x=bbox["x"], y=bbox["y"], width=bbox["width"], height=bbox["height"]),
        confidence=data.get("confidence"),
        class_id=data.get("class_id"),
        track_id=data.get("track_id"),
        centroid_x=data.get("centroid_x"),
        centroid_y=data.get("centroid_y"),
        pixel_count=data.get("pixel_count"),
    )


def convert_to_arrow(obj: Any) -> pa.Array:
    """Convert Retriever objects to a single-element Arrow JSON payload."""
    obj_type = type(obj)
    if obj_type in _arrow_converters:
        return _arrow_converters[obj_type](obj)

    if isinstance(obj, Action):
        return _pack(
            {
                "retriever_type": "Action",
                "type": obj.type,
                "parameters": obj.parameters,
                "timestamp": obj.timestamp,
                "priority": obj.priority,
            }
        )
    if isinstance(obj, Command):
        return _pack(
            {
                "retriever_type": "Command",
                "action": {
                    "type": obj.action.type,
                    "parameters": obj.action.parameters,
                    "timestamp": obj.action.timestamp,
                    "priority": obj.action.priority,
                },
                "robot_id": obj.robot_id,
                "expected_duration": obj.expected_duration,
                "timeout": obj.timeout,
            }
        )
    if isinstance(obj, Status):
        return _pack(
            {
                "retriever_type": "Status",
                "state": obj.state,
                "message": obj.message,
                "progress": obj.progress,
                "timestamp": obj.timestamp,
                "error_code": obj.error_code,
            }
        )
    if isinstance(obj, Timestamp):
        return _pack(
            {
                "retriever_type": "Timestamp",
                "seconds": obj.seconds,
                "nanoseconds": obj.nanoseconds,
            }
        )
    if isinstance(obj, ExecutionTimer):
        return _pack(
            {
                "retriever_type": "ExecutionTimer",
                "start_time": {
                    "seconds": obj.start_time.seconds,
                    "nanoseconds": obj.start_time.nanoseconds,
                },
                "expected_period": obj.expected_period,
                "actual_period": obj.actual_period,
                "iteration": obj.iteration,
            }
        )

    # Robotics typing v1 contract with stable identifiers for logging/replay.
    if isinstance(obj, Vector3):
        return _pack(
            {"retriever_type": "robotics.v1.Vector3", "x": obj.x, "y": obj.y, "z": obj.z}
        )
    if isinstance(obj, Quaternion):
        return _pack(
            {
                "retriever_type": "robotics.v1.Quaternion",
                "x": obj.x,
                "y": obj.y,
                "z": obj.z,
                "w": obj.w,
            }
        )
    if isinstance(obj, Header):
        return _pack(
            {
                "retriever_type": "robotics.v1.Header",
                "stamp_ns": obj.stamp_ns,
                "frame_id": obj.frame_id,
                "source": obj.source,
            }
        )
    if isinstance(obj, SE3Pose):
        return _pack(
            {
                "retriever_type": "robotics.v1.SE3Pose",
                "position": {"x": obj.position.x, "y": obj.position.y, "z": obj.position.z},
                "orientation": {
                    "x": obj.orientation.x,
                    "y": obj.orientation.y,
                    "z": obj.orientation.z,
                    "w": obj.orientation.w,
                },
            }
        )
    if isinstance(obj, Twist):
        return _pack(
            {
                "retriever_type": "robotics.v1.Twist",
                "linear": {"x": obj.linear.x, "y": obj.linear.y, "z": obj.linear.z},
                "angular": {"x": obj.angular.x, "y": obj.angular.y, "z": obj.angular.z},
            }
        )
    if isinstance(obj, Wrench):
        return _pack(
            {
                "retriever_type": "robotics.v1.Wrench",
                "force": {"x": obj.force.x, "y": obj.force.y, "z": obj.force.z},
                "torque": {"x": obj.torque.x, "y": obj.torque.y, "z": obj.torque.z},
            }
        )
    if isinstance(obj, PoseStamped):
        return _pack(
            {
                "retriever_type": "robotics.v1.PoseStamped",
                "header": {
                    "stamp_ns": obj.header.stamp_ns,
                    "frame_id": obj.header.frame_id,
                    "source": obj.header.source,
                },
                "pose": {
                    "position": {
                        "x": obj.pose.position.x,
                        "y": obj.pose.position.y,
                        "z": obj.pose.position.z,
                    },
                    "orientation": {
                        "x": obj.pose.orientation.x,
                        "y": obj.pose.orientation.y,
                        "z": obj.pose.orientation.z,
                        "w": obj.pose.orientation.w,
                    },
                },
            }
        )
    if isinstance(obj, TwistStamped):
        return _pack(
            {
                "retriever_type": "robotics.v1.TwistStamped",
                "header": {
                    "stamp_ns": obj.header.stamp_ns,
                    "frame_id": obj.header.frame_id,
                    "source": obj.header.source,
                },
                "twist": {
                    "linear": {
                        "x": obj.twist.linear.x,
                        "y": obj.twist.linear.y,
                        "z": obj.twist.linear.z,
                    },
                    "angular": {
                        "x": obj.twist.angular.x,
                        "y": obj.twist.angular.y,
                        "z": obj.twist.angular.z,
                    },
                },
            }
        )
    if isinstance(obj, WrenchStamped):
        return _pack(
            {
                "retriever_type": "robotics.v1.WrenchStamped",
                "header": {
                    "stamp_ns": obj.header.stamp_ns,
                    "frame_id": obj.header.frame_id,
                    "source": obj.header.source,
                },
                "wrench": {
                    "force": {
                        "x": obj.wrench.force.x,
                        "y": obj.wrench.force.y,
                        "z": obj.wrench.force.z,
                    },
                    "torque": {
                        "x": obj.wrench.torque.x,
                        "y": obj.wrench.torque.y,
                        "z": obj.wrench.torque.z,
                    },
                },
            }
        )
    if isinstance(obj, JointState):
        return _pack(
            {
                "retriever_type": "robotics.v1.JointState",
                "names": list(obj.names),
                "positions": list(obj.positions),
                "velocities": list(obj.velocities),
                "efforts": list(obj.efforts),
            }
        )

    # Perception typing standard (retriever.types.perception) with stable
    # identifiers, mirroring the robotics.v1 contract above.
    if isinstance(obj, Image2D):
        return _pack(
            {
                "retriever_type": "perception.v1.Image2D",
                "data": _encode_ndarray(obj.data),
                "encoding": obj.encoding,
                "header": _header_dict(obj.header),
                "frame_index": obj.frame_index,
            }
        )
    if isinstance(obj, BBox2D):
        return _pack(
            {
                "retriever_type": "perception.v1.BBox2D",
                "x": obj.x,
                "y": obj.y,
                "width": obj.width,
                "height": obj.height,
            }
        )
    if isinstance(obj, Detection2D):
        return _pack(
            {
                "retriever_type": "perception.v1.Detection2D",
                **_detection2d_dict(obj),
            }
        )
    if isinstance(obj, DetectionBatch):
        return _pack(
            {
                "retriever_type": "perception.v1.DetectionBatch",
                "detections": [_detection2d_dict(det) for det in obj.detections],
                "header": _header_dict(obj.header),
                "frame_index": obj.frame_index,
            }
        )
    if isinstance(obj, SegmentationMask2D):
        return _pack(
            {
                "retriever_type": "perception.v1.SegmentationMask2D",
                "mask": _encode_ndarray(obj.mask),
                "header": _header_dict(obj.header),
                "frame_index": obj.frame_index,
                "label_map": {str(k): v for k, v in (obj.label_map or {}).items()},
            }
        )
    if isinstance(obj, PointTarget2D):
        return _pack(
            {
                "retriever_type": "perception.v1.PointTarget2D",
                "label": obj.label,
                "x_norm": obj.x_norm,
                "y_norm": obj.y_norm,
                "confidence": obj.confidence,
                "header": _header_dict(obj.header),
                "frame_index": obj.frame_index,
            }
        )

    if isinstance(obj, np.ndarray):
        return pa.array(obj.ravel().tolist())
    if isinstance(obj, (list, tuple)):
        if obj and hasattr(obj[0], "_retriever_registered"):
            serialized = [convert_to_arrow(item)[0].as_py() for item in obj]
            return _pack(
                {
                    "type": "list",
                    "element_type": getattr(obj[0], "_retriever_type_name", type(obj[0]).__name__),
                    "data": serialized,
                }
            )
        return pa.array(list(obj))

    try:
        return _pack({"retriever_type": "json", "value": obj})
    except TypeError:
        encoded = base64.b64encode(pickle.dumps(obj)).decode("ascii")
        return _pack({"retriever_type": "pickle", "value": encoded})


def convert_from_arrow(arrow_array: pa.Array, target_type: Optional[Type] = None) -> Any:
    """Convert PyArrow array back to Retriever object payloads."""
    if target_type and target_type in _arrow_deserializers:
        return _arrow_deserializers[target_type](arrow_array)

    if not isinstance(arrow_array, pa.Array) or len(arrow_array) != 1:
        return arrow_array.to_numpy()

    raw = arrow_array[0].as_py()
    if not isinstance(raw, str):
        return raw

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    if not isinstance(data, dict):
        return data

    type_name = data.get("retriever_type")
    if not type_name:
        return data

    return _deserialize_by_type_name(type_name, data)


def _deserialize_by_type_name(type_name: str, data: dict[str, Any]) -> Any:
    if type_name == "Action":
        return Action(
            type=data["type"],
            parameters=data["parameters"],
            timestamp=data.get("timestamp"),
            priority=data.get("priority", 0),
        )
    if type_name == "Command":
        action_payload = data["action"]
        action = Action(
            type=action_payload["type"],
            parameters=action_payload["parameters"],
            timestamp=action_payload.get("timestamp"),
            priority=action_payload.get("priority", 0),
        )
        return Command(
            action=action,
            robot_id=data.get("robot_id", "default"),
            expected_duration=data.get("expected_duration"),
            timeout=data.get("timeout"),
        )
    if type_name == "Status":
        return Status(
            state=data["state"],
            message=data.get("message", ""),
            progress=data.get("progress"),
            timestamp=data.get("timestamp"),
            error_code=data.get("error_code"),
        )
    if type_name == "Timestamp":
        return Timestamp(seconds=data["seconds"], nanoseconds=data["nanoseconds"])
    if type_name == "ExecutionTimer":
        start = data["start_time"]
        return ExecutionTimer(
            start_time=Timestamp(seconds=start["seconds"], nanoseconds=start["nanoseconds"]),
            expected_period=data.get("expected_period"),
            actual_period=data.get("actual_period"),
            iteration=data.get("iteration", 0),
        )
    if type_name == "robotics.v1.Vector3":
        return Vector3(x=data["x"], y=data["y"], z=data["z"])
    if type_name == "robotics.v1.Quaternion":
        return Quaternion(x=data["x"], y=data["y"], z=data["z"], w=data["w"])
    if type_name == "robotics.v1.Header":
        return Header(
            stamp_ns=data["stamp_ns"],
            frame_id=data["frame_id"],
            source=data.get("source", "unknown"),
        )
    if type_name == "robotics.v1.SE3Pose":
        pos = data["position"]
        rot = data["orientation"]
        return SE3Pose(
            position=Vector3(x=pos["x"], y=pos["y"], z=pos["z"]),
            orientation=Quaternion(x=rot["x"], y=rot["y"], z=rot["z"], w=rot["w"]),
        )
    if type_name == "robotics.v1.Twist":
        linear = data["linear"]
        angular = data["angular"]
        return Twist(
            linear=Vector3(x=linear["x"], y=linear["y"], z=linear["z"]),
            angular=Vector3(x=angular["x"], y=angular["y"], z=angular["z"]),
        )
    if type_name == "robotics.v1.Wrench":
        force = data["force"]
        torque = data["torque"]
        return Wrench(
            force=Vector3(x=force["x"], y=force["y"], z=force["z"]),
            torque=Vector3(x=torque["x"], y=torque["y"], z=torque["z"]),
        )
    if type_name == "robotics.v1.PoseStamped":
        header_payload = data["header"]
        pose_payload = data["pose"]
        pos = pose_payload["position"]
        rot = pose_payload["orientation"]
        return PoseStamped(
            header=Header(
                stamp_ns=header_payload["stamp_ns"],
                frame_id=header_payload["frame_id"],
                source=header_payload.get("source", "unknown"),
            ),
            pose=SE3Pose(
                position=Vector3(x=pos["x"], y=pos["y"], z=pos["z"]),
                orientation=Quaternion(x=rot["x"], y=rot["y"], z=rot["z"], w=rot["w"]),
            ),
        )
    if type_name == "robotics.v1.TwistStamped":
        header_payload = data["header"]
        twist_payload = data["twist"]
        linear = twist_payload["linear"]
        angular = twist_payload["angular"]
        return TwistStamped(
            header=Header(
                stamp_ns=header_payload["stamp_ns"],
                frame_id=header_payload["frame_id"],
                source=header_payload.get("source", "unknown"),
            ),
            twist=Twist(
                linear=Vector3(x=linear["x"], y=linear["y"], z=linear["z"]),
                angular=Vector3(x=angular["x"], y=angular["y"], z=angular["z"]),
            ),
        )
    if type_name == "robotics.v1.WrenchStamped":
        header_payload = data["header"]
        wrench_payload = data["wrench"]
        force = wrench_payload["force"]
        torque = wrench_payload["torque"]
        return WrenchStamped(
            header=Header(
                stamp_ns=header_payload["stamp_ns"],
                frame_id=header_payload["frame_id"],
                source=header_payload.get("source", "unknown"),
            ),
            wrench=Wrench(
                force=Vector3(x=force["x"], y=force["y"], z=force["z"]),
                torque=Vector3(x=torque["x"], y=torque["y"], z=torque["z"]),
            ),
        )
    if type_name == "robotics.v1.JointState":
        return JointState(
            names=tuple(data["names"]),
            positions=tuple(data["positions"]),
            velocities=tuple(data["velocities"]),
            efforts=tuple(data["efforts"]),
        )
    if type_name == "perception.v1.Image2D":
        return Image2D(
            data=_decode_ndarray(data["data"]),
            encoding=data.get("encoding", "rgb8"),
            header=_header_from(data.get("header")),
            frame_index=data.get("frame_index"),
        )
    if type_name == "perception.v1.BBox2D":
        return BBox2D(x=data["x"], y=data["y"], width=data["width"], height=data["height"])
    if type_name == "perception.v1.Detection2D":
        return _detection2d_from(data)
    if type_name == "perception.v1.DetectionBatch":
        return DetectionBatch(
            detections=tuple(_detection2d_from(det) for det in data.get("detections", [])),
            header=_header_from(data.get("header")),
            frame_index=data.get("frame_index"),
        )
    if type_name == "perception.v1.SegmentationMask2D":
        return SegmentationMask2D(
            mask=_decode_ndarray(data["mask"]),
            header=_header_from(data.get("header")),
            frame_index=data.get("frame_index"),
            label_map={int(k): v for k, v in (data.get("label_map") or {}).items()},
        )
    if type_name == "perception.v1.PointTarget2D":
        return PointTarget2D(
            label=data.get("label"),
            x_norm=data.get("x_norm"),
            y_norm=data.get("y_norm"),
            confidence=data.get("confidence"),
            header=_header_from(data.get("header")),
            frame_index=data.get("frame_index"),
        )
    if type_name == "pickle":
        return pickle.loads(base64.b64decode(data["value"].encode("ascii")))
    if type_name == "json":
        return data.get("value")

    raise ValueError(f"Unknown type name: {type_name}")
