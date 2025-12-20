"""
Retriever Types System with External Registration

Like PyTorch's tensor system, this allows external packages to register
custom types that work seamlessly with Retriever flows and pipelines.

Core types provided by retriever-core:
- RGBImage, DepthImage, PointCloud
- Detection, BoundingBox, Pose3
- Action, Command, Status

External registration example:
    from retriever.types import register_type, Flow
    
    @register_type("MyCustomType")
    class MyCustomData:
        def __init__(self, data):
            self.data = data
    
    class MyFlow(Flow[RGBImage, MyCustomData]):
        def run(self, image: RGBImage) -> MyCustomData:
            return MyCustomData(process_image(image))
"""

from __future__ import annotations

from importlib import import_module

# Keep `retriever.types` lightweight: avoid importing optional heavy deps (e.g. numpy/torch)
# at import time. Types are loaded lazily on attribute access.
from .registry import register_type, get_registered_types, TypeRegistry, get_type, list_types, find_types

# Lazy conversion API to avoid importing heavy dependencies (pyarrow) unless needed
def convert_to_arrow(obj):
    from .conversions import convert_to_arrow as _cta
    return _cta(obj)

def convert_from_arrow(arr, target_type=None):
    from .conversions import convert_from_arrow as _cfa
    return _cfa(arr, target_type)

def register_conversion(type_class, to_arrow, from_arrow):
    from .conversions import register_conversion as _rc
    return _rc(type_class, to_arrow, from_arrow)

# -----------------------------------------------------------------------------
# Lazy type exports
# -----------------------------------------------------------------------------

_CORE_TYPES = {
    "RGBImage",
    "DepthImage",
    "PointCloud",
    "BoundingBox",
    "Detection",
    "Pose3",
    "Transform3",
    "Action",
    "Command",
    "Status",
    "Timestamp",
    "ExecutionTimer",
}

_VISION_TYPES = {
    "RGBDImage",
    "SegmentationMask",
    "EnvironmentObservation",
    "VLMResponse",
    "NLCommand",
    "WebResponse",
    "ActorHandle",
}

_ROBOTICS_TYPES = {
    "WorldState",
    "RobotState",
    "BeliefGraph",
    "Skill",
    "Plan",
    "StructuredPlan",
    "ActionPlan",
    "TaskGoal",
    "TaskInstance",
    "Trajectory",
    "ExecutionStatus",
    "ObjectVariable",
    "ObjectSymbol",
    "Observation",
    "ObservationHistory",
    "UnorderedObjectSet",
    "ObjectDescriptionDict",
}

_TYPE_MODULES = {
    **{name: "core_types" for name in _CORE_TYPES},
    **{name: "vision_types" for name in _VISION_TYPES},
    **{name: "robotics_types" for name in _ROBOTICS_TYPES},
}


def __getattr__(name: str):
    module = _TYPE_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    mod = import_module(f"{__name__}.{module}")
    return getattr(mod, name)


def __dir__():
    return sorted(set(globals().keys()) | set(_TYPE_MODULES.keys()))

# Export the main types for external use
__all__ = [
    # Most commonly used core types
    'RGBImage', 'Detection', 'BoundingBox', 'Pose3', 'Action',
    
    # Additional core types
    'DepthImage', 'PointCloud', 'Transform3', 'Command', 'Status',
    'Timestamp', 'ExecutionTimer',
    
    # Vision types
    'RGBDImage', 'SegmentationMask', 'EnvironmentObservation',
    'VLMResponse', 'NLCommand', 'WebResponse', 'ActorHandle',
    
    # Robotics types
    'WorldState', 'RobotState', 'BeliefGraph', 'Skill', 'Plan',
    'StructuredPlan', 'ActionPlan', 'TaskGoal', 'TaskInstance',
    'Trajectory', 'ExecutionStatus', 'ObjectVariable', 'ObjectSymbol',
    'Observation', 'ObservationHistory', 'UnorderedObjectSet', 'ObjectDescriptionDict',
    
    # Registry functions  
    'register_type', 'get_registered_types', 'get_type', 'list_types', 'find_types', 'TypeRegistry',
    'convert_to_arrow', 'convert_from_arrow', 'register_conversion',
]
