"""
Shared pipeline compositions for the Retriever framework.

This module contains pre-built pipeline compositions that combine multiple flows
to accomplish common robotics tasks. Pipelines provide higher-level abstractions
for complex multi-step operations.

Categories:
- perception: Vision and sensing pipelines
- manipulation: Object manipulation and grasping pipelines  
- navigation: Movement and path planning pipelines
- interaction: Multi-modal interaction pipelines
"""

from .perception import *  # noqa: F401, F403
from .manipulation import *  # noqa: F401, F403
from .navigation import *  # noqa: F401, F403
from .interaction import *  # noqa: F401, F403

__all__ = [
    # Re-export all from submodules
]