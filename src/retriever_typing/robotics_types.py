"""
Robotics-Specific Data Types

Types specifically for robotics applications including world state,
planning, and robot control.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple, Any

from retriever.types.spatial import SE3Pose

from .core_types import Action
from .registry import register_type


# ============================================================================
# World State and Belief Types
# ============================================================================

@register_type(description="World state representation")
@dataclass
class WorldState:
    """Represents the state of the world."""
    object_poses: Dict[str, SE3Pose]
    robot_pose: SE3Pose
    timestamp: Optional[float] = None


@register_type(description="Robot state information")
@dataclass
class RobotState:
    """Current robot state."""
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion
    objects_held: List[str] = None
    battery_level: float = 1.0
    last_error: Optional[str] = None
    
    def __post_init__(self):
        if self.objects_held is None:
            self.objects_held = []


@register_type(description="Probabilistic belief graph")
@dataclass
class BeliefGraph:
    """A probabilistic variant of the world state."""
    nodes: Set[str]
    edges: Dict[str, Set[str]]


# ============================================================================
# Planning and Task Types  
# ============================================================================

@register_type(description="Robot skill definition")
@dataclass
class Skill:
    """A skill that the robot can perform."""
    name: str
    params: Dict[str, Any]
    confidence: float = 1.0
    expected_duration: Optional[float] = None


@register_type(description="Sequence of skills")
@dataclass
class Plan:
    """A sequence of skills."""
    skills: List[Skill]
    confidence: float = 1.0
    
    @property
    def total_duration(self) -> Optional[float]:
        """Estimate total plan duration."""
        durations = [s.expected_duration for s in self.skills if s.expected_duration is not None]
        return sum(durations) if durations else None


@register_type(description="Structured plan representation")
@dataclass
class StructuredPlan:
    """A structured plan with typed steps."""
    steps: List[str]
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@register_type(description="Action sequence plan")
@dataclass  
class ActionPlan:
    """A sequence of actions for robot execution."""
    actions: List[Action]
    total_duration: Optional[float] = None
    
    @property
    def num_actions(self) -> int:
        return len(self.actions)


@register_type(description="Task goal specification")
@dataclass
class TaskGoal:
    """A task goal specification."""
    high_level_description: str
    affordances: Dict[str, Any]  # Available objects/actions
    success_criteria: Optional[str] = None


@register_type(description="Task instance")
@dataclass
class TaskInstance:
    """A specific instance of a task."""
    goal: TaskGoal
    initial_observation: Dict[str, Any]  # Using Dict instead of EnvironmentObservation to avoid circular imports
    seed: int = 0


# ============================================================================
# Trajectory and Motion Types
# ============================================================================

@register_type(description="3D trajectory")
@dataclass
class Trajectory:
    """A time-series of joint/base poses."""
    poses: List[SE3Pose]
    timestamps: Optional[List[float]] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get trajectory duration."""
        if self.timestamps and len(self.timestamps) >= 2:
            return self.timestamps[-1] - self.timestamps[0]
        return None


# ============================================================================
# Execution and Feedback Types
# ============================================================================

@register_type(description="Execution status")
@dataclass
class ExecutionStatus:
    """The status of an action's execution."""
    status: str  # SUCCESS / FAILURE / IN_PROGRESS
    metadata: Dict[str, Any]
    progress: Optional[float] = None  # 0.0 to 1.0
    error_message: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        return self.status in ["SUCCESS", "FAILURE"]


# ============================================================================
# Grounding and Symbolic Types
# ============================================================================

@register_type(description="Symbolic object variable")
@dataclass
class ObjectVariable:
    """A symbolic placeholder for an object."""
    name: str
    type_constraints: Optional[List[str]] = None  # e.g., ["graspable", "movable"]


@register_type(description="Grounded object symbol")
@dataclass
class ObjectSymbol:
    """A grounded object ID."""
    object_id: str
    object_type: Optional[str] = None
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


# ============================================================================
# Multi-sensor Observation Types
# ============================================================================

@register_type(description="Multi-sensor observation")
@dataclass
class Observation:
    """A collection of images from multiple sensors at a single timestep."""
    images: Dict[str, Any]  # e.g., {"front_camera": Image2D(...), "wrist_camera": Image2D(...)}
    timestamp: Optional[float] = None
    frame_id: str = "world"


@register_type(description="Observation history")
@dataclass
class ObservationHistory:
    """A deque of Observations."""
    history: List[Observation]
    max_length: int = 100
    
    def add(self, obs: Observation):
        """Add observation to history."""
        self.history.append(obs)
        if len(self.history) > self.max_length:
            self.history.pop(0)
    
    @property
    def latest(self) -> Optional[Observation]:
        """Get most recent observation."""
        return self.history[-1] if self.history else None


# ============================================================================
# Object and Scene Types
# ============================================================================

@register_type(description="Set of object IDs")
@dataclass
class UnorderedObjectSet:
    """A set of object IDs."""
    objects: Set[str]
    
    def add(self, obj_id: str):
        """Add object to set."""
        self.objects.add(obj_id)
    
    def remove(self, obj_id: str):
        """Remove object from set."""
        self.objects.discard(obj_id)


@register_type(description="Object description mapping")
@dataclass
class ObjectDescriptionDict:
    """A dictionary mapping object IDs to their descriptions."""
    descriptions: Dict[str, str]
    
    def add_object(self, obj_id: str, description: str):
        """Add object description."""
        self.descriptions[obj_id] = description