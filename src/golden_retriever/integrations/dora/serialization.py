from __future__ import annotations
"""
Apache Arrow Message Serialization for Dora Integration

This module provides zero-copy serialization utilities using Apache Arrow
for high-performance message passing in dora-rs dataflows.

Key Features:
- Zero-copy message serialization/deserialization
- Support for common robotics data types
- Efficient memory usage for large sensor data
- Type-safe Arrow schema generation
"""

from typing import Any, Dict, List, Optional, Type, Union
import io

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

import numpy as np
from dataclasses import dataclass


SchemaType = Any if not ARROW_AVAILABLE else pa.Schema

@dataclass
class ArrowMessage:
    """Container for Arrow-serialized messages with metadata."""
    data: bytes
    schema: SchemaType
    metadata: Dict[str, Any]


class ArrowMessageSerializer:
    """
    High-performance serializer using Apache Arrow for zero-copy message passing.
    
    This serializer converts Python objects to Arrow format for efficient
    transmission through dora-rs dataflows without copying large data structures.
    
    Example:
        ```python
        serializer = ArrowMessageSerializer()
        
        # Serialize image data
        image_data = np.random.uint8((480, 640, 3))
        message = serializer.serialize(image_data, "image")
        
        # Deserialize on the other side
        recovered_image = serializer.deserialize(message)
        ```
    """
    
    def __init__(self):
        """Initialize the Arrow serializer."""
        if not ARROW_AVAILABLE:
            raise ImportError(
                "PyArrow not available. Install with: pip install pyarrow"
            )
        
        # Cache for schema objects to avoid recreation
        self._schema_cache: Dict[str, pa.Schema] = {}
        
        # Register common robotics data type schemas
        self._register_common_schemas()
    
    def serialize(
        self, 
        data: Any, 
        data_type: str = "generic",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ArrowMessage:
        """
        Serialize data to Apache Arrow format for zero-copy transmission.
        
        Args:
            data: The data to serialize (numpy arrays, lists, dicts, etc.)
            data_type: Type hint for optimized serialization
            metadata: Additional metadata to include
            
        Returns:
            ArrowMessage containing serialized data and schema
        """
        metadata = metadata or {}
        
        # Convert data to Arrow format based on type
        arrow_data = self._convert_to_arrow(data, data_type)
        
        # Create Arrow table
        table = pa.table(arrow_data)
        
        # Add metadata to schema
        schema_metadata = {
            "data_type": data_type,
            "serializer_version": "1.0",
            **metadata
        }
        schema = table.schema.with_metadata(schema_metadata)
        table = table.cast(schema)
        
        # Serialize to bytes with IPC format (zero-copy friendly)
        sink = io.BytesIO()
        with ipc.new_stream(sink, schema) as writer:
            writer.write_table(table)
        
        return ArrowMessage(
            data=sink.getvalue(),
            schema=schema,
            metadata=metadata
        )
    
    def deserialize(self, message: ArrowMessage) -> Any:
        """
        Deserialize Arrow message back to Python objects.
        
        Args:
            message: The ArrowMessage to deserialize
            
        Returns:
            Original Python data structure
        """
        # Read Arrow table from bytes
        source = io.BytesIO(message.data)
        reader = ipc.open_stream(source)
        table = reader.read_all()
        
        # Extract data type from metadata
        schema_metadata = table.schema.metadata or {}
        data_type = schema_metadata.get(b"data_type", b"generic").decode()
        
        # Convert back to Python objects based on type
        return self._convert_from_arrow(table, data_type)
    
    def _convert_to_arrow(self, data: Any, data_type: str) -> Dict[str, pa.Array]:
        """Convert Python data to Arrow arrays."""
        if data_type == "image" and isinstance(data, np.ndarray):
            return self._serialize_image(data)
        elif data_type == "point_cloud" and isinstance(data, np.ndarray):
            return self._serialize_point_cloud(data)
        elif data_type == "pose" and isinstance(data, (list, tuple, np.ndarray)):
            return self._serialize_pose(data)
        elif data_type == "detections" and isinstance(data, list):
            return self._serialize_detections(data)
        elif isinstance(data, np.ndarray):
            return self._serialize_numpy_array(data)
        elif isinstance(data, dict):
            return self._serialize_dict(data)
        elif isinstance(data, (list, tuple)):
            return self._serialize_list(data)
        else:
            # Fallback: serialize as generic binary data
            return self._serialize_generic(data)
    
    def _convert_from_arrow(self, table: pa.Table, data_type: str) -> Any:
        """Convert Arrow table back to Python objects."""
        if data_type == "image":
            return self._deserialize_image(table)
        elif data_type == "point_cloud":
            return self._deserialize_point_cloud(table)
        elif data_type == "pose":
            return self._deserialize_pose(table)
        elif data_type == "detections":
            return self._deserialize_detections(table)
        elif data_type == "numpy_array":
            return self._deserialize_numpy_array(table)
        elif data_type == "dict":
            return self._deserialize_dict(table)
        elif data_type == "list":
            return self._deserialize_list(table)
        else:
            return self._deserialize_generic(table)
    
    def _serialize_image(self, image: np.ndarray) -> Dict[str, pa.Array]:
        """Serialize image data efficiently."""
        return {
            "height": pa.array([image.shape[0]]),
            "width": pa.array([image.shape[1]]),
            "channels": pa.array([image.shape[2] if len(image.shape) > 2 else 1]),
            "dtype": pa.array([str(image.dtype)]),
            "data": pa.array([image.tobytes()])
        }
    
    def _deserialize_image(self, table: pa.Table) -> np.ndarray:
        """Deserialize image data."""
        height = table["height"][0].as_py()
        width = table["width"][0].as_py()
        channels = table["channels"][0].as_py()
        dtype = np.dtype(table["dtype"][0].as_py())
        data_bytes = table["data"][0].as_py()
        
        shape = (height, width, channels) if channels > 1 else (height, width)
        return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    
    def _serialize_point_cloud(self, points: np.ndarray) -> Dict[str, pa.Array]:
        """Serialize 3D point cloud data."""
        return {
            "num_points": pa.array([points.shape[0]]),
            "num_features": pa.array([points.shape[1]]),
            "dtype": pa.array([str(points.dtype)]),
            "data": pa.array([points.tobytes()])
        }
    
    def _deserialize_point_cloud(self, table: pa.Table) -> np.ndarray:
        """Deserialize 3D point cloud data."""
        num_points = table["num_points"][0].as_py()
        num_features = table["num_features"][0].as_py()
        dtype = np.dtype(table["dtype"][0].as_py())
        data_bytes = table["data"][0].as_py()
        
        return np.frombuffer(data_bytes, dtype=dtype).reshape(num_points, num_features)
    
    def _serialize_pose(self, pose: Union[list, tuple, np.ndarray]) -> Dict[str, pa.Array]:
        """Serialize pose data (position + orientation)."""
        pose_array = np.array(pose, dtype=np.float64)
        return {
            "pose_data": pa.array(pose_array.tolist()),
            "format": pa.array(["xyzquat"])  # x,y,z,qx,qy,qz,qw format
        }
    
    def _deserialize_pose(self, table: pa.Table) -> np.ndarray:
        """Deserialize pose data."""
        return np.array(table["pose_data"].to_pylist())
    
    def _serialize_detections(self, detections: List[Dict]) -> Dict[str, pa.Array]:
        """Serialize object detection results."""
        if not detections:
            return {"num_detections": pa.array([0])}
        
        # Extract fields from detection dictionaries
        boxes = []
        scores = []
        labels = []
        
        for det in detections:
            boxes.append(det.get("box", [0, 0, 0, 0]))
            scores.append(det.get("score", 0.0))
            labels.append(det.get("label", ""))
        
        return {
            "num_detections": pa.array([len(detections)]),
            "boxes": pa.array([item for sublist in boxes for item in sublist]),
            "scores": pa.array(scores),
            "labels": pa.array(labels)
        }
    
    def _deserialize_detections(self, table: pa.Table) -> List[Dict]:
        """Deserialize object detection results."""
        num_dets = table["num_detections"][0].as_py()
        if num_dets == 0:
            return []
        
        boxes_flat = table["boxes"].to_pylist()
        scores = table["scores"].to_pylist()
        labels = table["labels"].to_pylist()
        
        # Reshape boxes from flat list
        boxes = [boxes_flat[i*4:(i+1)*4] for i in range(num_dets)]
        
        return [
            {"box": box, "score": score, "label": label}
            for box, score, label in zip(boxes, scores, labels)
        ]
    
    def _serialize_numpy_array(self, array: np.ndarray) -> Dict[str, pa.Array]:
        """Generic numpy array serialization."""
        return {
            "shape": pa.array(list(array.shape)),
            "dtype": pa.array([str(array.dtype)]),
            "data": pa.array([array.tobytes()])
        }
    
    def _deserialize_numpy_array(self, table: pa.Table) -> np.ndarray:
        """Generic numpy array deserialization."""
        shape = tuple(table["shape"].to_pylist())
        dtype = np.dtype(table["dtype"][0].as_py())
        data_bytes = table["data"][0].as_py()
        
        return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    
    def _serialize_dict(self, data: Dict) -> Dict[str, pa.Array]:
        """Serialize dictionary data."""
        keys = list(data.keys())
        values = [str(v) for v in data.values()]  # Convert all to strings for simplicity
        
        return {
            "keys": pa.array(keys),
            "values": pa.array(values)
        }
    
    def _deserialize_dict(self, table: pa.Table) -> Dict:
        """Deserialize dictionary data."""
        keys = table["keys"].to_pylist()
        values = table["values"].to_pylist()
        return dict(zip(keys, values))
    
    def _serialize_list(self, data: List) -> Dict[str, pa.Array]:
        """Serialize list data."""
        # Convert all items to strings for simplicity
        string_items = [str(item) for item in data]
        return {
            "items": pa.array(string_items),
            "length": pa.array([len(data)])
        }
    
    def _deserialize_list(self, table: pa.Table) -> List:
        """Deserialize list data."""
        return table["items"].to_pylist()
    
    def _serialize_generic(self, data: Any) -> Dict[str, pa.Array]:
        """Fallback serialization using pickle."""
        import pickle
        return {
            "pickle_data": pa.array([pickle.dumps(data)])
        }
    
    def _deserialize_generic(self, table: pa.Table) -> Any:
        """Fallback deserialization using pickle."""
        import pickle
        return pickle.loads(table["pickle_data"][0].as_py())
    
    def _register_common_schemas(self) -> None:
        """Pre-register common schemas for performance."""
        # Image schema
        self._schema_cache["image"] = pa.schema([
            ("height", pa.int32()),
            ("width", pa.int32()),
            ("channels", pa.int32()),
            ("dtype", pa.string()),
            ("data", pa.binary())
        ])
        
        # Point cloud schema
        self._schema_cache["point_cloud"] = pa.schema([
            ("num_points", pa.int32()),
            ("num_features", pa.int32()),
            ("dtype", pa.string()),
            ("data", pa.binary())
        ])
        
        # Pose schema
        self._schema_cache["pose"] = pa.schema([
            ("pose_data", pa.list_(pa.float64())),
            ("format", pa.string())
        ])
    
    def get_schema(self, data_type: str) -> Optional[pa.Schema]:
        """Get pre-computed schema for a data type."""
        return self._schema_cache.get(data_type)
