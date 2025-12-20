#!/usr/bin/env python3
"""
016_vlm_planning.py - Verification of Symbolic Planning Migration

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

from retriever.flow import Flow, Pipeline, flow_io
from retriever.flow.clock import Trigger
from retriever.types import SkillSignature, GroundedSkill
from retriever.types import Object, Type

# ======================= TYPES =======================

@dataclass
class RGBImage:
    data: str  # Mock data

@dataclass
class Action:
    name: str
    target: str

@flow_io
@dataclass
class ObjectList:
    items: List[Object]

@flow_io
@dataclass
class ActionList:
    items: List[Action]

@flow_io
@dataclass
class Result:
    status: str

# Use the migrated symbolic types
ObjectT = Type("Entity", ["x", "y", "z"])

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
    
    def run(self, objects: ObjectList) -> ActionList:
        print(f"🧠 Planner: Reasoning about {len(objects.items)} objects...")
        # Verify GroundedSkill usage
        actions = []
        for obj in objects.items:
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
        p_handle = perception_flow @ Trigger(on="tick")
        pl_handle = planner_flow @ Trigger(on="perception")
        ex_handle = executor_flow @ Trigger(on="planner")
        
        # Connect them using experimental operator support or .then()
        p_handle >> pl_handle >> ex_handle
    
    print("\nRunning pipeline logic manually...")
    obj_list = perception_flow.run(None)
    action_list = planner_flow.run(obj_list)
    result = executor_flow.run(action_list)
    
    print(f"\n✅ Result: {result}")

if __name__ == "__main__":
    main()
