"""
Typed Oracle Example: Modern bilevel planning with foundation models.

This example demonstrates how to use the updated Retriever type system for
robot task planning and execution. It combines:
- Structured type definitions for environment observations and planning
- Foundation model integration (VLMs, segmentation models) via Ray
- Bilevel planning with high-level VLM reasoning and low-level skills
- Type-safe pipeline composition using the Module protocol

Key improvements over the original example:
1. Strong typing for all data structures
2. Clean separation of concerns with typed interfaces  
3. Reusable pipeline components
4. Better error handling and debugging
5. Integration with the Flow/Eff framework for composable workflows
"""

import os
import pathlib
import random
from typing import List, Dict, Any, Optional
from dataclasses import asdict

import numpy as np
import ray
import torch
from PIL import Image
from rich import print

# Core Retriever types
from retriever.core.types import (
    Module,
    RGBImage,
    EnvironmentObservation,
    TaskGoal,
    TaskInstance,
    VLMResponse,
    StructuredPlan,
    Detection,
    BoundingBox,
    ActorHandle,
    ExecutionStatus,
)

# Environment and task imports (keeping existing for compatibility)
from retriever.envs.ravens import tasks
from retriever.envs.ravens.envs.environment import Environment
from retriever.models import utils
from retriever.models.segmentation.langsam_actor import LangSAM
from retriever.planners.examples.planning_helper import (
    format_and_simplify_plan,
    process_image_and_question,
)


# ###################### Typed Environment Wrapper #######################

class TypedEnvironmentWrapper:
    """Type-safe wrapper for the Ravens environment."""
    
    def __init__(self, env: Environment):
        self.env = env
        
    def reset(self) -> EnvironmentObservation:
        """Reset environment and return typed observation."""
        raw_obs = self.env.reset()
        return EnvironmentObservation(
            color=raw_obs["color"],
            depth=raw_obs.get("depth", []),
            metadata=self.env.info
        )
    
    def step(self, action: Any) -> tuple[EnvironmentObservation, float, bool, Dict[str, Any]]:
        """Execute action and return typed observation."""
        raw_obs, reward, done, info = self.env.step(action)
        
        obs = EnvironmentObservation(
            color=raw_obs["color"],
            depth=raw_obs.get("depth", []),
            metadata=info
        )
        
        return obs, reward, done, info
    
    def get_task_goal(self) -> TaskGoal:
        """Get current task goal in typed format."""
        return TaskGoal(
            high_level_description=self.env.info["high_level_lang_goal"],
            affordances=self.env.info.get("blockbowl_affordance", {})
        )


# ###################### Typed Planning Modules ##########################

class VLMPlannerModule:
    """Vision-Language Model planner with typed interfaces."""
    
    def __init__(self):
        pass
    
    def __call__(self, task_instance: TaskInstance) -> VLMResponse:
        """Generate plan using VLM given task instance."""
        
        # Extract images for VLM processing
        front_obs = task_instance.initial_observation.color[0]  # front camera
        topdown_obs = task_instance.initial_observation.color[3]  # topdown camera
        
        # Save images for VLM processing
        front_img = Image.fromarray(front_obs)
        topdown_img = Image.fromarray(topdown_obs)
        
        utils.save_image(front_img, "front_observation.png")
        utils.save_image(topdown_img, "topdown_observation.png")
        
        # Generate planning prompt
        affordances = task_instance.goal.affordances
        block_keys = [k for k in affordances.keys() if "block" in k]
        bowl_keys = [k for k in affordances.keys() if "bowl" in k]
        
        planning_prompt = self._create_planning_prompt(
            task_instance.goal.high_level_description,
            block_keys,
            bowl_keys
        )
        
        # Call VLM API
        response = process_image_and_question(
            planning_prompt,
            "prompt1.png",  # Few-shot examples
            "prompt2.png",
            "prompt3.png", 
            "topdown_observation.png",
            question=task_instance.goal.high_level_description
        )
        
        return VLMResponse(
            content=response['choices'][0]['message']['content'],
            metadata={"model": "gpt-4v", "prompt": planning_prompt}
        )
    
    def _create_planning_prompt(self, task_goal: str, blocks: List[str], bowls: List[str]) -> str:
        """Create structured planning prompt."""
        
        block_desc = ", ".join(blocks)
        bowl_desc = ", ".join(bowls)
        
        return f"""
        Given the task described, observe the environment where objects are placed within boxes 
        for clearer identification. Generate a plan using only [pick up] and [place it] skills.

        Task Description: {task_goal}

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


class PlanFormatterModule:
    """Module to format VLM responses into structured plans."""
    
    def __call__(self, response: VLMResponse) -> StructuredPlan:
        """Format VLM response into structured plan."""
        
        # Use existing formatting logic
        formatted_steps = format_and_simplify_plan({"choices": [{"message": {"content": response.content}}]})
        
        return StructuredPlan(
            steps=formatted_steps,
            confidence=1.0  # Could be extracted from response metadata
        )


class SegmentationModule:
    """Module for object segmentation using LangSAM."""
    
    def __init__(self, langsam_actor: ActorHandle):
        self.actor = langsam_actor
    
    def __call__(self, image: RGBImage, text_prompt: str) -> List[Detection]:
        """Segment objects in image using text prompt."""
        
        # Convert to PIL for actor
        pil_image = Image.fromarray(image.data)
        
        # Call segmentation actor
        masks, boxes, phrases, logits = ray.get(
            self.actor.handle.predict.remote(pil_image, text_prompt)
        )
        
        # Convert to typed detections
        detections = []
        for i, (box, phrase, logit) in enumerate(zip(boxes, phrases, logits)):
            # Assuming box format is [x, y, w, h]
            detection = Detection(
                label=phrase,
                confidence=float(logit),
                bbox=BoundingBox(x=box[0], y=box[1], width=box[2], height=box[3])
            )
            detections.append(detection)
        
        return detections


# ###################### Typed Execution Pipeline #########################

class TypedOracleExecutor:
    """Type-safe oracle executor for bilevel planning."""
    
    def __init__(self, env_wrapper: TypedEnvironmentWrapper, oracle_agent: Any):
        self.env = env_wrapper
        self.oracle = oracle_agent
        self.vlm_planner = VLMPlannerModule()
        self.plan_formatter = PlanFormatterModule()
        
    def execute_task_instance(self, task_instance: TaskInstance, save_video: bool = False) -> ExecutionStatus:
        """Execute a single task instance using bilevel planning."""
        
        print(f"Executing task: {task_instance.goal.high_level_description}")
        
        # Reset environment with seed
        np.random.seed(task_instance.seed)
        random.seed(task_instance.seed)
        self.env.env.seed(task_instance.seed)
        
        # Get initial observation
        obs = self.env.reset()
        task_instance.initial_observation = obs
        
        if save_video:
            self.env.env.start_rec(f"oracle-typed-seed{task_instance.seed}")
        
        try:
            # High-level planning with VLM
            vlm_response = self.vlm_planner(task_instance)
            print(f"VLM Response: {vlm_response.content}")
            
            # Format into structured plan
            structured_plan = self.plan_formatter(vlm_response)
            print(f"Structured Plan: {structured_plan.steps}")
            
            # Execute plan with oracle agent
            success_count = 0
            for step_idx, instruction in enumerate(structured_plan.steps):
                print(f"Executing step {step_idx + 1}: {instruction}")
                
                # Use oracle agent for low-level execution
                action = self.oracle.act(obs, instruction)
                obs, reward, done, info = self.env.step(action)
                
                success = info.get("success", False)
                print(f"Step success: {success}")
                
                if success:
                    success_count += 1
                    
                # Update observation for next step
                obs = EnvironmentObservation(
                    color=obs["color"] if isinstance(obs, dict) else obs.color,
                    depth=obs.get("depth", []) if isinstance(obs, dict) else obs.depth,
                    metadata=info
                )
                
            final_success = success_count > 0  # Could be more sophisticated
            
            return ExecutionStatus(
                status="SUCCESS" if final_success else "FAILURE",
                metadata={
                    "steps_completed": success_count,
                    "total_steps": len(structured_plan.steps),
                    "vlm_model": vlm_response.metadata.get("model", "unknown")
                }
            )
            
        except Exception as e:
            print(f"Execution failed: {e}")
            return ExecutionStatus(
                status="FAILURE", 
                metadata={"error": str(e)}
            )
        
        finally:
            if save_video:
                self.env.env.end_rec()


# ###################### Main Execution Function ##########################

def main():
    """Main execution function demonstrating typed oracle planning."""
    
    # Setup Ray and GPU configuration
    use_gpu = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    print(f"GPU available: {use_gpu}, Number of GPUs: {num_gpus}")
    
    ray.init(num_gpus=0 if not use_gpu else num_gpus)
    print("Ray resources:", ray.available_resources())
    
    # Initialize Ray actors
    actor_options = {"num_gpus": 0.1} if use_gpu else {}
    langsam_actor_handle = ActorHandle(
        handle=LangSAM.options(**actor_options).remote(use_gpu=use_gpu),
        name="langsam"
    )
    
    # Setup environment
    root_dir = pathlib.Path.cwd().parent.parent
    assets_root = os.path.join(root_dir, "envs/ravens/envs/assets/")
    
    record_cfg = {
        "save_video": True,
        "save_video_path": "./tmp/images",
        "add_text": True,
        "fps": 20,
        "video_height": 640,
        "video_width": 720,
    }
    
    env = Environment(
        assets_root, disp=True, shared_memory=False, hz=480, record_cfg=record_cfg
    )
    
    typed_env = TypedEnvironmentWrapper(env)
    
    # Setup task
    task_list = [
        "stack-blocks",
        "put-blocks-matching-colors", 
        "sort-primary-color-blocks",
    ]
    
    task_name = task_list[0]
    task = tasks.names[task_name]()
    task.mode = "test"
    
    print(f"Task: {task_name}")
    print(f"Task Goal: {task.get_lang_goal()}")
    
    # Initialize oracle agent
    oracle_agent = task.step_oracle(env)
    
    # Create executor
    executor = TypedOracleExecutor(typed_env, oracle_agent)
    
    # Run evaluation
    n_eval = 3
    base_seed = 8888
    results = []
    
    for i in range(n_eval):
        print(f"\n=== Evaluation {i + 1}/{n_eval} ===")
        
        # Set task in environment
        env.set_task(task)
        
        # Create task instance
        task_instance = TaskInstance(
            goal=TaskGoal(
                high_level_description=task.get_lang_goal(),
                affordances={}  # Will be populated after reset
            ),
            initial_observation=EnvironmentObservation(color=[], depth=[], metadata={}),
            seed=base_seed + i * 2
        )
        
        # Execute task
        result = executor.execute_task_instance(task_instance, save_video=True)
        results.append(result)
        
        print(f"Result: {result.status}")
        print(f"Metadata: {result.metadata}")
    
    # Print summary
    successes = sum(1 for r in results if r.status == "SUCCESS")
    print(f"\nFinal Results: {successes}/{n_eval} ({successes/n_eval:.3f} success rate)")
    
    for i, result in enumerate(results):
        print(f"Trial {i+1}: {result.status} - {result.metadata}")


if __name__ == "__main__":
    # Check API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or not openai_key.startswith("sk-"):
        raise ValueError("OpenAI API key not found or invalid")
    
    main()