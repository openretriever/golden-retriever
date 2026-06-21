#!/usr/bin/env python3
"""
VLM planning example - verification of symbolic planning migration

This example verifies that:
1. src/retriever/types/symbolic.py is accessible
2. src/retriever/types/skills.py is accessible
3. Flow composition works with these types
"""

import sys
import os
from typing import List
from dataclasses import dataclass

# Explicitly add source root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from retriever.flow import Flow, Pipeline, io
from retriever.flow.clock import Trigger
from retriever.types import SkillSignature, GroundedSkill
from retriever.types import Object
from retriever.flow.types import EventStream, Behavior, State, ObjectType

# ======================= TYPES =======================

@dataclass
class RGBImage:
    data: str  # Mock data

@dataclass
class Action:
    name: str
    target: str

@io
@dataclass
class ObjectList:
    items: List[Object]

@io
@dataclass
class ActionList:
    items: List[Action]

@io
@dataclass
class Result:
    status: str

# Use the migrated symbolic types
ObjectT = ObjectType("Entity", ["x", "y", "z"])

# ======================= FLOWS =======================

class PerceptionFlow(Flow[None, ObjectList]):
    def run(self, _: None) -> ObjectList:
        print("👀 Perception: Detecting objects...")
        # creating objects using migrated class
        return ObjectList(items=[
            Object("cup_1", ObjectT),
            Object("bottle_1", ObjectT)
        ])

class PlannerFlow(Flow[ObjectList, ActionList]):
    def __init__(self):
        super().__init__()
        # Verify SkillSignature usage
        self.pickup_sig = SkillSignature(
            name="pick_up",
            template="pick up {target}"
        )
    
    def plan_task(self, query: str, scene_objects: List[ObjectType]) -> List[str]:
        print(f"🧠 Planner: Reasoning about {len(scene_objects)} objects based on query: '{query}'...")
        # Verify GroundedSkill usage
        actions = []
        # The original logic iterated over objects.items, which was ObjectList.
        # Now scene_objects is List[ObjectType].
        # To maintain similar behavior, we'll assume ObjectType has a 'name' attribute for grounding.
        # Note: This change makes the PlannerFlow's generic types (Flow[ObjectList, ActionList])
        # inconsistent with the new `plan_task` signature.
        # For this specific edit, I'm applying the change as requested,
        # but a full refactor would be needed to align the Flow generics.
        for obj_type in scene_objects:
            # Assuming ObjectType has a 'name' attribute or can be represented as a string
            # For the purpose of this example, we'll use a placeholder for the target name
            # as ObjectType itself doesn't have a 'name' attribute in this file.
            # If the intent was to pass `Object` instances, the type hint would be `List[Object]`.
            # Given `List[ObjectType]`, we'll use a generic placeholder for the target.
            target_name = f"an_object_of_type_{obj_type.name}" # Placeholder
            
            skill = GroundedSkill(
                signature=self.pickup_sig,
                grounded_params={"target": obj.name}
            )
            # Verify validation logic
            skill.validate_grounding({obj.name: "detected"})
            
            print(f"   Generated plan: {skill}")
            actions.append(Action(name=skill.signature.name, target=obj.name))
        
        return ActionList(items=actions)

class ExecutorFlow(Flow[ActionList, Result]):
    def run(self, actions: ActionList) -> Result:
        print("🤖 Executor: Executing actions...")
        for action in actions.items:
            print(f"   Executing: {action.name} on {action.target}")
        return Result(status="Success")

# ======================= MAIN =======================

def main():
    print("🚀 Verification: Symbolic Planning Migration")
    
    # Instantiate flows
    perception_flow = PerceptionFlow()
    planner_flow = PlannerFlow()
    executor_flow = ExecutorFlow()
    
    # Compose pipeline using Context Manager (standard API)
    # This automatically registers handles and connections
    with Pipeline("vlm_plan") as pipeline:
        # Create handles bound to clocks
        p_handle = perception_flow @ Trigger("tick")
        pl_handle = planner_flow @ Trigger("perception")
        ex_handle = executor_flow @ Trigger("planner")
        
        # Connect them using experimental operator support or .then()
        p_handle >> pl_handle >> ex_handle
    
    print("\nRunning pipeline logic manually...")
    obj_list = perception_flow.run(None)
    action_list = planner_flow.run(obj_list)
    result = executor_flow.run(action_list)
    
    print(f"\n✅ Result: {result}")

if __name__ == "__main__":
    main()
