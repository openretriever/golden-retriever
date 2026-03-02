"""Retriever type system with dual-surface robotics typing exports.

Primary access:
    from golden_retriever.robotics_typing import PoseStamped

Compatibility access:
    from golden_retriever.types import PoseStamped

Registry access:
    from golden_retriever.types import get_type
    PoseStamped = get_type("PoseStamped")
"""

from __future__ import annotations

from importlib import import_module

from .registry import (
    TypeRegistry,
    find_types,
    get_registered_types,
    get_type,
    list_types,
    register_type,
)


def convert_to_arrow(obj):
    from .conversions import convert_to_arrow as _cta

    return _cta(obj)


def convert_from_arrow(arr, target_type=None):
    from .conversions import convert_from_arrow as _cfa

    return _cfa(arr, target_type)


def register_conversion(type_class, to_arrow, from_arrow):
    from .conversions import register_conversion as _rc

    return _rc(type_class, to_arrow, from_arrow)


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

_ROBOTICS_TYPING_V1_TYPES = {
    "Header",
    "Vector3",
    "Quaternion",
    "SE3Pose",
    "PoseStamped",
    "Twist",
    "TwistStamped",
    "Wrench",
    "WrenchStamped",
    "JointState",
    "validate_header",
    "validate_quaternion",
    "validate_pose_stamped",
    "validate_joint_state",
}

_ROOT_PACKAGE = __name__.split(".", 1)[0]
_TYPE_MODULES = {
    **{name: "core_types" for name in _CORE_TYPES},
    **{name: "vision_types" for name in _VISION_TYPES},
    **{name: "robotics_types" for name in _ROBOTICS_TYPES},
    **{
        name: f"{_ROOT_PACKAGE}.robotics_typing.v1"
        for name in _ROBOTICS_TYPING_V1_TYPES
    },
}


def __getattr__(name: str):
    module = _TYPE_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    if "." in module:
        mod = import_module(module)
    else:
        mod = import_module(f"{__name__}.{module}")
    return getattr(mod, name)


def __dir__():
    return sorted(set(globals().keys()) | set(_TYPE_MODULES.keys()))


__all__ = [
    "RGBImage",
    "Detection",
    "BoundingBox",
    "Action",
    "DepthImage",
    "PointCloud",
    "Command",
    "Status",
    "Timestamp",
    "ExecutionTimer",
    "Pose3",
    "Transform3",
    "RGBDImage",
    "SegmentationMask",
    "EnvironmentObservation",
    "VLMResponse",
    "NLCommand",
    "WebResponse",
    "ActorHandle",
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
    "Header",
    "Vector3",
    "Quaternion",
    "SE3Pose",
    "PoseStamped",
    "Twist",
    "TwistStamped",
    "Wrench",
    "WrenchStamped",
    "JointState",
    "validate_header",
    "validate_quaternion",
    "validate_pose_stamped",
    "validate_joint_state",
    "register_type",
    "get_registered_types",
    "get_type",
    "list_types",
    "find_types",
    "TypeRegistry",
    "convert_to_arrow",
    "convert_from_arrow",
    "register_conversion",
]
