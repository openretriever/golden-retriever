"""
Flow-Based Oracle Example: Demonstrating Module/Flow/Pipeline Design

This example shows how to use the Retriever Flow framework for composable robotics
pipelines. It demonstrates the key architectural patterns:

1. **Module Protocol**: All components implement Module[I, O] for type-safe composition
2. **Flow Combinators**: Use .then() and .fanout() to build complex pipelines  
3. **Pipeline Composition**: Build reusable, testable components that compose cleanly
4. **Eff Monad**: Handle stateful robot operations with clean functional composition
5. **Parallel Processing**: Use fanout() for multi-sensor, multi-hypothesis workflows

Key architectural advantages:
- Type safety catches pipeline mismatches at development time
- Components are easily testable in isolation
- Reusable modules work across different robots and tasks
- Easy parallelization with fanout() for sensor fusion
- Clean separation between pure computation and stateful effects
"""

import os
import sys
import pathlib
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add project root to Python path
project_root = pathlib.Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import ray
import torch
from rich import print

# Core Retriever framework
from retriever.core.types import (
    Module,
    Eff,
    pure,
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
from retriever.core.flow import Flow

# Shared modules from perception and planning
from retriever.perception import (
    ObservationExtractor,
    ImageSaver,
    ObjectDetector,
    create_perception_pipeline,
)
from retriever.planning import (
    PromptGenerator,
    VLMCaller,
    PlanFormatter,
    RobotExecutor,
    RobotState,
    create_planning_pipeline,
    create_full_pipeline,
)

# Environment imports (keeping existing for compatibility)
from retriever.envs.ravens import tasks
from retriever.envs.ravens.envs.environment import Environment
from retriever.models.segmentation.langsam_actor import LangSAM


# ###################### Environment-Specific Components ##################


# ###################### Main Execution System ############################

class FlowBasedOracleSystem:
    """
    Complete oracle system using Flow-based pipeline architecture.
    
    Demonstrates how to combine:
    - Pure computation pipelines (Flow)
    - Stateful robot operations (Eff)
    - Type-safe composition (Module protocol)
    """
    
    def __init__(self, env: Environment, oracle_agent: Any, langsam_actor: ActorHandle):
        self.env = env
        self.oracle = oracle_agent
        self.robot_executor = RobotExecutor(oracle_agent)
        
        # Create reusable pipelines using shared modules
        self.perception_pipeline = create_perception_pipeline(langsam_actor)
        self.planning_pipeline = create_planning_pipeline()
        self.full_pipeline = create_full_pipeline(langsam_actor)
    
    def execute_task_instance(self, task_instance: TaskInstance) -> ExecutionStatus:
        """Execute task using Flow-based pipelines and Eff monads."""
        
        print(f"Executing task with Flow pipelines: {task_instance.goal.high_level_description}")
        
        # Setup environment
        np.random.seed(task_instance.seed)
        random.seed(task_instance.seed)
        self.env.seed(task_instance.seed)
        
        # Get initial observation (simplified - would integrate with env reset)
        raw_obs = self.env.reset()
        observation = EnvironmentObservation(
            color=raw_obs["color"],
            depth=raw_obs.get("depth", []),
            metadata=self.env.info
        )
        
        try:
            # Execute pure computation pipeline
            print("🔄 Running perception + planning pipeline...")
            plan, detections = self.full_pipeline((observation, task_instance.goal))
            
            print(f"📋 Generated plan: {plan.steps}")
            print(f"👁️  Detected objects: {[d.label for d in detections]}")
            
            # Execute stateful robot operations using Eff monad
            print("🤖 Executing plan with robot...")
            initial_robot_state = RobotState()
            
            execution_result, final_robot_state = self.robot_executor.execute_plan(plan).run(initial_robot_state)
            
            print(f"✅ Execution completed: {execution_result.status}")
            print(f"🎯 Final robot state: {final_robot_state}")
            
            return execution_result
            
        except Exception as e:
            print(f"❌ Pipeline execution failed: {e}")
            return ExecutionStatus(
                status="FAILURE",
                metadata={"error": str(e), "pipeline": "flow_based"}
            )


# ###################### Demo Runner ######################################

def main():
    """Demonstrate Flow-based robotics pipeline architecture."""
    
    print("🚀 Starting Flow-based Oracle Example")
    print("=" * 60)
    
    # Setup Ray and actors
    use_gpu = torch.cuda.is_available()
    ray.init(num_gpus=0 if not use_gpu else torch.cuda.device_count())
    
    actor_options = {"num_gpus": 0.1} if use_gpu else {}
    langsam_actor = ActorHandle(
        handle=LangSAM.options(**actor_options).remote(use_gpu=use_gpu),
        name="langsam"
    )
    
    # Setup environment
    root_dir = pathlib.Path.cwd().parent.parent
    assets_root = os.path.join(root_dir, "envs/ravens/envs/assets/")
    
    env = Environment(assets_root, disp=False, shared_memory=False, hz=480)
    
    # Setup task
    task_name = "stack-blocks"
    task = tasks.names[task_name]()
    task.mode = "test"
    env.set_task(task)
    
    oracle_agent = task.step_oracle(env)
    
    print(f"📋 Task: {task_name}")
    print(f"🎯 Goal: {task.get_lang_goal()}")
    
    # Create Flow-based system
    system = FlowBasedOracleSystem(env, oracle_agent, langsam_actor)
    
    # Demonstrate individual pipeline components
    print("\n🔧 Testing Individual Pipeline Components:")
    print("-" * 40)
    
    # Test perception pipeline
    print("1. Testing Perception Pipeline...")
    dummy_obs = EnvironmentObservation(
        color=[np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)],
        depth=[],
        metadata={}
    )
    
    try:
        perception_result = system.perception_pipeline(dummy_obs)
        print(f"   ✓ Perception pipeline works: {type(perception_result)}")
    except Exception as e:
        print(f"   ✗ Perception pipeline failed: {e}")
    
    # Test planning pipeline
    print("2. Testing Planning Pipeline...")
    dummy_goal = TaskGoal(
        high_level_description="stack the blocks",
        affordances={"red_block": {}, "blue_block": {}, "green_bowl": {}}
    )
    
    try:
        planning_result = system.planning_pipeline(dummy_goal)
        print(f"   ✓ Planning pipeline works: {planning_result.steps[:2]}...")
    except Exception as e:
        print(f"   ✗ Planning pipeline failed: {e}")
    
    # Test Eff monad operations
    print("3. Testing Eff Monad Operations...")
    dummy_plan = StructuredPlan(steps=["pick up red block", "place on blue block"])
    initial_state = RobotState()
    
    try:
        result, final_state = system.robot_executor.execute_plan(dummy_plan).run(initial_state)
        print(f"   ✓ Eff execution works: {result.status}, final state: {final_state.success_count} successes")
    except Exception as e:
        print(f"   ✗ Eff execution failed: {e}")
    
    # Run full integrated example
    print("\n🎬 Running Full Integrated Example:")
    print("-" * 40)
    
    task_instance = TaskInstance(
        goal=TaskGoal(
            high_level_description=task.get_lang_goal(),
            affordances={"red_block": {}, "blue_block": {}, "bowl": {}}
        ),
        initial_observation=EnvironmentObservation(color=[], depth=[], metadata={}),
        seed=42
    )
    
    final_result = system.execute_task_instance(task_instance)
    
    print(f"\n🏁 Final Result: {final_result.status}")
    print(f"📊 Metadata: {final_result.metadata}")
    
    print("\n✨ Flow-based Oracle Example Complete!")
    print("=" * 60)
    
    # Demonstrate key architectural benefits
    print("\n🏗️  Key Architectural Benefits Demonstrated:")
    print("   ✓ Type-safe pipeline composition with Flow.then() and Flow.fanout()")
    print("   ✓ Reusable modules implementing Module[I, O] protocol") 
    print("   ✓ Clean separation of pure computation (Flow) and stateful effects (Eff)")
    print("   ✓ Parallel processing with fanout() for multi-sensor workflows")
    print("   ✓ Easy testing and mocking of individual pipeline components")
    print("   ✓ Composable architecture that scales from simple to complex systems")


if __name__ == "__main__":
    # Check API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or not openai_key.startswith("sk-"):
        print("⚠️  Warning: OpenAI API key not found - VLM calls will fail")
        print("   Set OPENAI_API_KEY environment variable for full functionality")
    
    main()