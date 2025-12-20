"""
Combined perception and planning pipelines for the Retriever framework.

This module provides factory functions to create integrated pipelines that
combine perception and planning components.
"""

from typing import List, Tuple

from retriever.core.types import (
    EnvironmentObservation,
    TaskGoal,
    StructuredPlan,
    Detection,
    ActorHandle,
)
from retriever.core.flow import Flow
from retriever.perception.pipelines import create_perception_pipeline
from .pipelines import create_planning_pipeline


def create_full_pipeline(langsam_actor: ActorHandle) -> Flow[Tuple[EnvironmentObservation, TaskGoal], Tuple[StructuredPlan, List[Detection]]]:
    """
    Create a full perception + planning pipeline using Flow composition.
    
    Demonstrates how to combine multiple pipelines with fanout().
    """
    
    perception_pipeline = create_perception_pipeline(langsam_actor)
    planning_pipeline = create_planning_pipeline()
    
    # Extract components from input tuple
    extract_obs = Flow.from_module(lambda x: x[0])  # Extract observation
    extract_goal = Flow.from_module(lambda x: x[1])  # Extract goal
    
    # Combine pipelines in parallel
    combined_pipeline = (
        extract_obs.then(perception_pipeline)  # Obs → (paths, detections)
        .fanout(extract_goal.then(planning_pipeline))  # Goal → Plan
        .then(Flow.from_module(lambda x: (x[1], x[0][1])))  # Reorder to (plan, detections)
    )
    
    return combined_pipeline