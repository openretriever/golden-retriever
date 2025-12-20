"""
Planning pipeline factories for the Retriever framework.

This module provides factory functions to create reusable planning pipelines
using Flow composition.
"""

from typing import Tuple

from retriever.core.types import (
    TaskGoal,
    StructuredPlan,
)
from retriever.core.flow import Flow
from .modules import PromptGenerator, VLMCaller, PlanFormatter


def create_planning_pipeline() -> Flow[TaskGoal, StructuredPlan]:
    """
    Create a planning pipeline using Flow composition.
    
    Pipeline: TaskGoal → Generate Prompt → Call VLM → Format Plan
    """
    
    generate_prompt = Flow.from_module(PromptGenerator())
    call_vlm = Flow.from_module(VLMCaller())
    format_plan = Flow.from_module(PlanFormatter())
    
    # Helper to combine prompt with saved image paths (would need context in real implementation)
    def combine_with_images(prompt: str) -> Tuple[str, Tuple[str, str]]:
        # In real implementation, this would access saved image paths from context
        return prompt, ("front_observation.png", "topdown_observation.png")
    
    combine_inputs = Flow.from_module(combine_with_images)
    
    # Compose the planning pipeline
    pipeline = (
        generate_prompt
        .then(combine_inputs)
        .then(call_vlm)
        .then(format_plan)
    )
    
    return pipeline