"""
Applied action / status types for the GoldenRetriever type pack.

These are the applied-tier payloads that have no standard equivalent in the
runtime. Perception payloads live in `retriever.types.perception` and spatial
ones in `retriever.types.spatial` — never redefine a standard type here.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

from .registry import register_type


# ============================================================================
# Robot Action and Control Types
# ============================================================================

@register_type(description="Robot action command")
@dataclass
class Action:
    """Generic robot action."""
    type: str  # e.g., "move", "grasp", "release"
    parameters: dict  # Action-specific parameters
    timestamp: Optional[float] = None
    priority: int = 0  # Higher priority = more urgent
    
    def __post_init__(self):
        """Ensure parameters is a dict."""
        if not isinstance(self.parameters, dict):
            raise ValueError("Action parameters must be a dict")
    
    def to_arrow(self) -> dict:
        """Convert to Arrow-compatible format."""
        import json
        return {
            "type": self.type,
            "parameters": json.dumps(self.parameters),
            "timestamp": self.timestamp,
            "priority": self.priority
        }
    
    @classmethod
    def from_arrow(cls, arrow_data: dict) -> 'Action':
        """Convert from Arrow format."""
        import json
        return cls(
            type=arrow_data["type"],
            parameters=json.loads(arrow_data["parameters"]),
            timestamp=arrow_data.get("timestamp"),
            priority=arrow_data.get("priority", 0)
        )


@register_type(description="Robot command with execution info")
@dataclass
class Command:
    """Robot command with execution metadata."""
    action: Action
    robot_id: str = "default"
    expected_duration: Optional[float] = None  # Expected execution time in seconds
    timeout: Optional[float] = None  # Maximum execution time
    
    @property
    def action_type(self) -> str:
        """Get the action type."""
        return self.action.type
    
    def to_arrow(self) -> dict:
        """Convert to Arrow-compatible format."""
        return {
            "action": self.action.to_arrow(),
            "robot_id": self.robot_id,
            "expected_duration": self.expected_duration,
            "timeout": self.timeout
        }
    
    @classmethod
    def from_arrow(cls, arrow_data: dict) -> 'Command':
        """Convert from Arrow format."""
        return cls(
            action=Action.from_arrow(arrow_data["action"]),
            robot_id=arrow_data.get("robot_id", "default"),
            expected_duration=arrow_data.get("expected_duration"),
            timeout=arrow_data.get("timeout")
        )


StatusState = Literal["pending", "running", "completed", "failed", "cancelled"]


@register_type(description="Execution status information")
@dataclass
class Status:
    """Execution status."""
    state: StatusState
    message: str = ""
    progress: Optional[float] = None  # Progress percentage (0-100)
    timestamp: Optional[float] = None
    error_code: Optional[int] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if execution is complete (success or failure)."""
        return self.state in ["completed", "failed", "cancelled"]
    
    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.state == "completed"


# ============================================================================
# Temporal and Execution Types
# ============================================================================

@register_type(description="High-precision timestamp")
@dataclass
class Timestamp:
    """High-precision timestamp."""
    seconds: int
    nanoseconds: int
    
    @classmethod
    def now(cls) -> 'Timestamp':
        """Create timestamp for current time."""
        import time
        t = time.time()
        return cls(seconds=int(t), nanoseconds=int((t % 1) * 1e9))
    
    def to_float(self) -> float:
        """Convert to floating-point seconds."""
        return self.seconds + self.nanoseconds / 1e9


@register_type(description="Execution timing information")
@dataclass
class ExecutionTimer:
    """Execution timing information."""
    start_time: Timestamp
    expected_period: Optional[float] = None  # Expected period in seconds
    actual_period: Optional[float] = None    # Actual period in seconds
    iteration: int = 0
    
    @property
    def is_timing_violation(self) -> bool:
        """Check if there's a timing violation."""
        if self.expected_period is None or self.actual_period is None:
            return False
        return self.actual_period > self.expected_period * 1.5  # 50% tolerance
