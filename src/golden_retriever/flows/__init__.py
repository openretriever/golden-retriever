"""
Shared flow components for the Retriever framework.

This module contains reusable flow components that can be composed into pipelines.
Flows are designed to be atomic, composable, and type-safe building blocks.

Categories:
- vision: Computer vision and perception flows
- control: Robot control and actuation flows
- reasoning: Planning and decision-making flows
- sensing: Sensor processing and data acquisition flows
"""

# Flows module - use direct imports for better clarity and explicit dependencies
# Examples:
#   from retriever.flows.vision.camera import CameraFlow
#   from retriever.flows.vision.detection import YOLOFlow  
#   from retriever.flows.vision.visualization import ColorDetector
#   from retriever.flows.control.arm import ArmControlFlow  # when implemented
#   from retriever.flows.reasoning.planning import PlanningFlow  # when implemented