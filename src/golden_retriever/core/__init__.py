"""
Core components of the Retriever framework.

This module provides the foundational abstractions for building composable,
type-safe robotics pipelines with distributed execution capabilities.
"""

from .flow import Flow # Arrow Backward compatibility
from .types import Eff, pure, Pipeline
from .execution import ExecutionEngine

__all__ = [
    "Flow",  # Core flow abstraction
    "Eff", "pure",    # Stateful computation monad
    "Pipeline", "ExecutionEngine",  # Execution framework
]
