"""
Reusable planning modules for the Retriever framework.

This module contains composable planning components that implement the Module[I, O]
protocol for type-safe pipeline composition.
"""

import random
from typing import Tuple, Any
from dataclasses import dataclass

from retriever.core.types import (
    Module,
    Eff,
    TaskGoal,
    VLMResponse,
    StructuredPlan,
    ExecutionStatus,
)
from retriever.planning.examples.planning_helper import (
    format_and_simplify_plan,
    process_image_and_question,
)


class PromptGenerator:
    """Module to generate planning prompts from task goals."""
    
    def __call__(self, task_goal: TaskGoal) -> str:
        """Generate structured planning prompt."""
        affordances = task_goal.affordances
        block_keys = [k for k in affordances.keys() if "block" in k]
        bowl_keys = [k for k in affordances.keys() if "bowl" in k]
        
        block_desc = ", ".join(block_keys)
        bowl_desc = ", ".join(bowl_keys)
        
        return f"""
        Given the task described, observe the environment where objects are placed within boxes 
        for clearer identification. Generate a plan using only [pick up] and [place it] skills.

        Task Description: {task_goal.high_level_description}

        Instructions:
        1. Identify all available objects and their colors
        2. Generate step-by-step plan following the format
        3. Do not involve interaction with the "table"

        Available Objects:
        Blocks: {block_desc}
        Bowls: {bowl_desc}

        Plan Format:
        - "pick up the [color/object]"
        - "and place it on the [color/object]"
        """


class VLMCaller:
    """Module to call VLM API with images and prompt."""
    
    def __call__(self, inputs: Tuple[str, Tuple[str, str]]) -> VLMResponse:
        """Call VLM with prompt and image paths."""
        prompt, (front_path, topdown_path) = inputs
        
        # Call VLM API (using existing helper)
        response = process_image_and_question(
            prompt,
            "prompt1.png",  # Few-shot examples
            "prompt2.png",
            "prompt3.png",
            topdown_path,
            question=prompt  # Could extract from prompt
        )
        
        return VLMResponse(
            content=response['choices'][0]['message']['content'],
            metadata={"model": "gpt-4v", "prompt": prompt}
        )


class PlanFormatter:
    """Module to format VLM responses into structured plans."""
    
    def __call__(self, response: VLMResponse) -> StructuredPlan:
        """Format VLM response into structured plan."""
        # Use existing formatting logic
        formatted_steps = format_and_simplify_plan({
            "choices": [{"message": {"content": response.content}}]
        })
        
        return StructuredPlan(
            steps=formatted_steps,
            confidence=1.0
        )


@dataclass
class RobotState:
    """State tracking for robot operations."""
    position: str = "unknown"
    last_action: str = "none"
    success_count: int = 0


class RobotExecutor:
    """Stateful robot execution using Eff monad."""
    
    def __init__(self, oracle_agent: Any):
        self.oracle = oracle_agent
    
    def execute_step(self, instruction: str) -> Eff[RobotState, bool]:
        """Execute a single plan step, returning success status."""
        
        def run_step(state: RobotState) -> Tuple[bool, RobotState]:
            try:
                # This would need the current observation - simplified for demo
                # action = self.oracle.act(current_obs, instruction)
                # obs, reward, done, info = env.step(action)
                # success = info.get("success", False)
                
                # Simulated for demonstration
                success = random.random() > 0.3  # 70% success rate
                
                new_state = RobotState(
                    position=f"after_{instruction}",
                    last_action=instruction,
                    success_count=state.success_count + (1 if success else 0)
                )
                
                return success, new_state
                
            except Exception as e:
                print(f"Execution error: {e}")
                return False, state
        
        return Eff(run_step)
    
    def execute_plan(self, plan: StructuredPlan) -> Eff[RobotState, ExecutionStatus]:
        """Execute full plan using Eff monad composition."""
        
        def run_plan(initial_state: RobotState) -> Tuple[ExecutionStatus, RobotState]:
            current_state = initial_state
            total_successes = 0
            
            for step_idx, instruction in enumerate(plan.steps):
                print(f"Executing step {step_idx + 1}: {instruction}")
                
                # Execute step and update state
                success, current_state = self.execute_step(instruction).run(current_state)
                
                if success:
                    total_successes += 1
                    print(f"✓ Step {step_idx + 1} succeeded")
                else:
                    print(f"✗ Step {step_idx + 1} failed")
            
            final_success = total_successes > 0
            
            status = ExecutionStatus(
                status="SUCCESS" if final_success else "FAILURE",
                metadata={
                    "steps_completed": total_successes,
                    "total_steps": len(plan.steps),
                    "final_position": current_state.position
                }
            )
            
            return status, current_state
        
        return Eff(run_plan)