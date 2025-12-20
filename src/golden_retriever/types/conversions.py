"""
Type Conversion System for Dora/PyArrow Integration

Provides automatic conversion between Retriever types and PyArrow arrays
for efficient data transfer in Dora dataflows.
"""

import pyarrow as pa
import numpy as np
from typing import Any, Dict, Callable, Optional, Type
import json
import pickle
import base64

from .registry import _global_registry, TypeInfo
from .core_types import *

# Registry for custom converters
_arrow_converters: Dict[Type, Callable[[Any], pa.Array]] = {}
_arrow_deserializers: Dict[Type, Callable[[pa.Array], Any]] = {}


def register_conversion(type_class: Type, 
                       to_arrow: Callable[[Any], pa.Array],
                       from_arrow: Callable[[pa.Array], Any]):
    """Register custom PyArrow conversion for a type."""
    _arrow_converters[type_class] = to_arrow
    _arrow_deserializers[type_class] = from_arrow


def convert_to_arrow(obj: Any) -> pa.Array:
    """
    Convert any Retriever type to PyArrow array.
    
    This is the main function used by Dora operators to send data.
    """
    obj_type = type(obj)
    
    # Check for custom converter first
    if obj_type in _arrow_converters:
        return _arrow_converters[obj_type](obj)
    
    # Check for built-in conversions
    if isinstance(obj, RGBImage):
        return _rgbimage_to_arrow(obj)
    elif isinstance(obj, DepthImage):
        return _depthimage_to_arrow(obj)
    elif isinstance(obj, PointCloud):
        return _pointcloud_to_arrow(obj)
    elif isinstance(obj, Detection):
        return _detection_to_arrow(obj)
    elif isinstance(obj, BoundingBox):
        return _bbox_to_arrow(obj)
    elif isinstance(obj, Pose3):
        return _pose3_to_arrow(obj)
    elif isinstance(obj, Transform3):
        return _transform3_to_arrow(obj)
    elif isinstance(obj, Action):
        return _action_to_arrow(obj)
    elif isinstance(obj, Command):
        return _command_to_arrow(obj)
    elif isinstance(obj, Status):
        return _status_to_arrow(obj)
    elif isinstance(obj, Timestamp):
        return _timestamp_to_arrow(obj)
    elif isinstance(obj, ExecutionTimer):
        return _timer_to_arrow(obj)
    elif isinstance(obj, np.ndarray):
        return pa.array(obj.ravel())
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0 and hasattr(obj[0], '_retriever_registered'):
            # List of registered types - serialize each
            serialized = [convert_to_arrow(item) for item in obj]
            # Pack as JSON with type info
            data = {
                'type': 'list',
                'element_type': obj[0]._retriever_type_name,
                'data': [_arrow_to_dict(arr) for arr in serialized]
            }
            return pa.array([json.dumps(data)])
        else:
            return pa.array(obj)
    else:
        # Fallback: serialize as JSON
        try:
            json_str = json.dumps(obj, default=str)
            return pa.array([json_str])
        except (TypeError, ValueError):
            # Final fallback: pickle and base64 encode
            pickled = pickle.dumps(obj)
            encoded = base64.b64encode(pickled).decode('ascii')
            return pa.array([encoded])


def convert_from_arrow(arrow_array: pa.Array, target_type: Optional[Type] = None) -> Any:
    """
    Convert PyArrow array back to Retriever type.
    
    This is used by Dora operators to receive data.
    """
    if target_type and target_type in _arrow_deserializers:
        return _arrow_deserializers[target_type](arrow_array)
    
    # Try to infer type from arrow data structure
    if isinstance(arrow_array, pa.Array) and len(arrow_array) == 1:
        # Single-element array might contain JSON metadata
        try:
            data_str = arrow_array[0].as_py()
            if isinstance(data_str, str):
                data = json.loads(data_str)
                if isinstance(data, dict) and 'retriever_type' in data:
                    type_name = data['retriever_type']
                    return _deserialize_by_type_name(type_name, data)
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Fallback: return numpy array
    return arrow_array.to_numpy()


# ============================================================================
# Built-in Type Converters
# ============================================================================

def _rgbimage_to_arrow(img: RGBImage) -> pa.Array:
    """Convert RGBImage to PyArrow array."""
    data = {
        'retriever_type': 'RGBImage',
        'data': img.data.tobytes(),
        'shape': img.data.shape,
        'dtype': str(img.data.dtype),
        'timestamp': img.timestamp,
        'camera_id': img.camera_id
    }
    return pa.array([json.dumps(data)])


def _depthimage_to_arrow(img: DepthImage) -> pa.Array:
    """Convert DepthImage to PyArrow array."""
    data = {
        'retriever_type': 'DepthImage', 
        'data': img.data.tobytes(),
        'shape': img.data.shape,
        'dtype': str(img.data.dtype),
        'timestamp': img.timestamp,
        'camera_id': img.camera_id
    }
    return pa.array([json.dumps(data)])


def _pointcloud_to_arrow(pc: PointCloud) -> pa.Array:
    """Convert PointCloud to PyArrow array."""
    data = {
        'retriever_type': 'PointCloud',
        'points': pc.points.tobytes(),
        'points_shape': pc.points.shape,
        'points_dtype': str(pc.points.dtype),
        'colors': pc.colors.tobytes() if pc.colors is not None else None,
        'colors_shape': pc.colors.shape if pc.colors is not None else None,
        'colors_dtype': str(pc.colors.dtype) if pc.colors is not None else None,
        'timestamp': pc.timestamp,
        'frame_id': pc.frame_id
    }
    return pa.array([json.dumps(data)])


def _detection_to_arrow(det: Detection) -> pa.Array:
    """Convert Detection to PyArrow array."""
    data = {
        'retriever_type': 'Detection',
        'label': det.label,
        'confidence': det.confidence,
        'bbox': {
            'x': det.bbox.x,
            'y': det.bbox.y, 
            'width': det.bbox.width,
            'height': det.bbox.height
        },
        'mask': det.mask.tobytes() if det.mask is not None else None,
        'mask_shape': det.mask.shape if det.mask is not None else None,
        'features': det.features.tobytes() if det.features is not None else None,
        'features_shape': det.features.shape if det.features is not None else None
    }
    return pa.array([json.dumps(data)])


def _bbox_to_arrow(bbox: BoundingBox) -> pa.Array:
    """Convert BoundingBox to PyArrow array."""
    data = {
        'retriever_type': 'BoundingBox',
        'x': bbox.x,
        'y': bbox.y,
        'width': bbox.width, 
        'height': bbox.height
    }
    return pa.array([json.dumps(data)])


def _pose3_to_arrow(pose: Pose3) -> pa.Array:
    """Convert Pose3 to PyArrow array."""
    data = {
        'retriever_type': 'Pose3',
        'position': pose.position.tolist(),
        'orientation': pose.orientation.tolist(),
        'frame_id': pose.frame_id
    }
    return pa.array([json.dumps(data)])


def _transform3_to_arrow(transform: Transform3) -> pa.Array:
    """Convert Transform3 to PyArrow array.""" 
    data = {
        'retriever_type': 'Transform3',
        'matrix': transform.matrix.tolist(),
        'from_frame': transform.from_frame,
        'to_frame': transform.to_frame
    }
    return pa.array([json.dumps(data)])


def _action_to_arrow(action: Action) -> pa.Array:
    """Convert Action to PyArrow array."""
    data = {
        'retriever_type': 'Action',
        'type': action.type,
        'parameters': action.parameters,
        'timestamp': action.timestamp,
        'priority': action.priority
    }
    return pa.array([json.dumps(data)])


def _command_to_arrow(cmd: Command) -> pa.Array:
    """Convert Command to PyArrow array."""
    data = {
        'retriever_type': 'Command',
        'action': {
            'type': cmd.action.type,
            'parameters': cmd.action.parameters,
            'timestamp': cmd.action.timestamp,
            'priority': cmd.action.priority
        },
        'robot_id': cmd.robot_id,
        'expected_duration': cmd.expected_duration,
        'timeout': cmd.timeout
    }
    return pa.array([json.dumps(data)])


def _status_to_arrow(status: Status) -> pa.Array:
    """Convert Status to PyArrow array."""
    data = {
        'retriever_type': 'Status',
        'state': status.state,
        'message': status.message,
        'progress': status.progress,
        'timestamp': status.timestamp,
        'error_code': status.error_code
    }
    return pa.array([json.dumps(data)])


def _timestamp_to_arrow(ts: Timestamp) -> pa.Array:
    """Convert Timestamp to PyArrow array."""
    data = {
        'retriever_type': 'Timestamp',
        'seconds': ts.seconds,
        'nanoseconds': ts.nanoseconds
    }
    return pa.array([json.dumps(data)])


def _timer_to_arrow(timer: ExecutionTimer) -> pa.Array:
    """Convert ExecutionTimer to PyArrow array."""
    data = {
        'retriever_type': 'ExecutionTimer',
        'start_time': {
            'seconds': timer.start_time.seconds,
            'nanoseconds': timer.start_time.nanoseconds
        },
        'expected_period': timer.expected_period,
        'actual_period': timer.actual_period,
        'iteration': timer.iteration
    }
    return pa.array([json.dumps(data)])


# ============================================================================
# Deserialization Helpers
# ============================================================================

def _deserialize_by_type_name(type_name: str, data: dict) -> Any:
    """Deserialize object by registered type name."""
    if type_name == 'RGBImage':
        return RGBImage(
            data=np.frombuffer(data['data'], dtype=data['dtype']).reshape(data['shape']),
            timestamp=data['timestamp'],
            camera_id=data['camera_id']
        )
    elif type_name == 'DepthImage':
        return DepthImage(
            data=np.frombuffer(data['data'], dtype=data['dtype']).reshape(data['shape']),
            timestamp=data['timestamp'], 
            camera_id=data['camera_id']
        )
    elif type_name == 'Detection':
        bbox_data = data['bbox']
        bbox = BoundingBox(
            x=bbox_data['x'], y=bbox_data['y'],
            width=bbox_data['width'], height=bbox_data['height']
        )
        mask = np.frombuffer(data['mask']).reshape(data['mask_shape']) if data['mask'] else None
        features = np.frombuffer(data['features']).reshape(data['features_shape']) if data['features'] else None
        
        return Detection(
            label=data['label'],
            confidence=data['confidence'],
            bbox=bbox,
            mask=mask,
            features=features
        )
    # Add other type deserializers as needed...
    else:
        raise ValueError(f"Unknown type name: {type_name}")


def _arrow_to_dict(arr: pa.Array) -> dict:
    """Convert PyArrow array to dict for serialization."""
    return {
        'type': str(arr.type),
        'data': arr.to_pylist()
    }