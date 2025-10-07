#!/usr/bin/env python3
"""
Libero Mock Test

Test script that runs mock policies on real Libero environments.
Based on external/openpi/examples/libero/main.py but uses our controller Flow system.

This demonstrates:
1. Loading real Libero tasks and environments
2. Using our RobotObservation/RobotAction types
3. Running mock controllers for testing
4. Integration with the full pipeline
"""

import sys
import os
import logging
import pathlib
import dataclasses
from typing import Dict, Any, Optional
import numpy as np
import imageio
import tqdm

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from openpi_controller.types import RobotObservation, RobotAction, libero_obs_to_robot_obs, robot_action_to_libero_action
    from openpi_controller.flows import MockControllerFlow, RandomControllerFlow, OpenPIControllerFlow
except ImportError:
    from robotics_types import RobotObservation, RobotAction, libero_obs_to_robot_obs, robot_action_to_libero_action
    from controller_flow import MockControllerFlow, RandomControllerFlow, OpenPIControllerFlow

# Import Libero
LIBERO_AVAILABLE = True
try:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as e:
    LIBERO_AVAILABLE = False
    benchmark = None
    get_libero_path = None
    OffScreenRenderEnv = None
    print(f"⚠️  Libero not available: {e}")
    print("💡 Install Libero: cd retriever/envs/libero && bash install.sh")


@dataclasses.dataclass
class TestConfig:
    """Configuration for mock testing"""
    task_suite_name: str = "libero_90"  # Task suite to test
    num_trials_per_task: int = 3        # Number of trials per task
    num_tasks: int = 3                  # Number of tasks to test
    max_steps: int = 200                # Max steps per episode
    num_steps_wait: int = 10            # Steps to wait for sim stabilization
    video_out_path: str = "examples/openpi/videos/mock_test"  # Video output
    seed: int = 42                      # Random seed
    controller_type: str = "mock"       # Controller type: mock, random, openpi
    save_videos: bool = True            # Save episode videos
    
    # OpenPI-specific settings (if using openpi controller)
    openpi_checkpoint_dir: str = "/mnt/arc/yygx/pkgs_baselines/openpi/checkpoints/pi0_fast_libero_low_mem_finetune_yy/baseline_default/29999"
    openpi_config_name: str = "pi0_fast_libero_low_mem_finetune"


def create_controller(config: TestConfig):
    """Create controller based on config"""
    
    if config.controller_type == "mock":
        print("🎭 Creating mock controller")
        return MockControllerFlow(
            n_joints=6,  # Libero uses 6-DOF end-effector control
            movement_amplitude=0.05,  # Small movements
            movement_frequency=0.02   # Slow movement
        )
    
    elif config.controller_type == "random":
        print("🎲 Creating random controller")
        return RandomControllerFlow(
            n_joints=6,
            action_scale=0.03,  # Small random actions
            seed=config.seed
        )
    
    elif config.controller_type == "openpi":
        print("🤖 Creating OpenPI controller")

        return OpenPIControllerFlow(
            checkpoint_dir=config.openpi_checkpoint_dir,
            config_name=config.openpi_config_name
        )
    
    else:
        raise ValueError(f"Unknown controller type: {config.controller_type}")



def get_libero_env_and_task(task, resolution: int = 256, seed: int = 42):
    """Create Libero environment for a task (based on main.py)"""
    
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    
    # Check if BDDL file exists
    if not os.path.exists(task_bddl_file):
        raise FileNotFoundError(f"BDDL file not found: {task_bddl_file}")
    
    # Create environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution
    }
    
    print(f"🔧 Creating OffScreenRenderEnv with args: {env_args}")
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    
    return env, task_description


def run_single_episode(controller, env, task_description: str, initial_state, config: TestConfig) -> Dict[str, Any]:
    """Run a single episode with the controller"""
    
    # Reset environment and set initial state
    env.reset()
    libero_obs = env.set_init_state(initial_state)
    
    # Episode tracking
    episode_data = {
        "success": False,
        "steps": 0,
        "total_reward": 0.0,
        "images": []  # For video recording
    }
    
    # Libero dummy action for initial stabilization
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    
    t = 0
    while t < config.max_steps + config.num_steps_wait:
        try:
            # Wait for simulation to stabilize (important for Libero)
            if t < config.num_steps_wait:
                libero_obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue
            
            # Convert Libero observation to our standard format
            robot_obs = libero_obs_to_robot_obs(libero_obs, task_description)
            
            # Save image for video if enabled
            if config.save_videos and "agentview" in robot_obs.images:
                episode_data["images"].append(robot_obs.images["agentview"])
            
            # Get action from controller
            robot_action = controller(robot_obs)
            
            # Convert to Libero format and execute
            libero_action = robot_action_to_libero_action(robot_action)
            libero_obs, reward, done, info = env.step(libero_action)
            
            # Update episode data
            episode_data["total_reward"] += reward
            episode_data["steps"] = t - config.num_steps_wait + 1
            
            # Check for success
            if done:
                episode_data["success"] = True
                logging.info(f"✅ Episode succeeded at step {episode_data['steps']}")
                break
            
            t += 1
            
        except Exception as e:
            logging.error(f"❌ Error at step {t}: {e}")
            break
    
    if not episode_data["success"] and t >= config.max_steps + config.num_steps_wait:
        logging.info(f"⏰ Episode timed out after {config.max_steps} steps")
    
    return episode_data


def save_episode_video(episode_data: Dict[str, Any], task_name: str, trial_idx: int, config: TestConfig):
    """Save episode video"""
    
    if not config.save_videos or not episode_data["images"]:
        return
    
    # Create output directory
    pathlib.Path(config.video_out_path).mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    success_str = "success" if episode_data["success"] else "failure"
    task_clean = task_name.replace(" ", "_").replace("/", "_")
    filename = f"{config.controller_type}_{task_clean}_trial{trial_idx}_{success_str}.mp4"
    video_path = pathlib.Path(config.video_out_path) / filename
    
    # Save video
    try:
        imageio.mimwrite(
            video_path,
            episode_data["images"],
            fps=10
        )
        logging.info(f"📹 Video saved: {video_path}")
    except Exception as e:
        logging.warning(f"⚠️  Failed to save video: {e}")


def run_task_evaluation(controller, task_suite, task_id: int, config: TestConfig) -> Dict[str, Any]:
    """Run evaluation on a single task"""
    
    # Get task and initial states
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    
    logging.info(f"\n🎯 Task {task_id}: {task.name}")
    logging.info(f"   Description: {task.language}")
    
    # Create environment
    try:
        env, task_description = get_libero_env_and_task(task, resolution=256, seed=config.seed)
    except FileNotFoundError as e:
        logging.error(f"❌ Task setup failed: {e}")
        return {"task_id": task_id, "success_rate": 0.0, "error": str(e)}
    
    # Run trials
    task_results = []
    successes = 0
    
    for trial_idx in range(min(config.num_trials_per_task, len(initial_states))):
        logging.info(f"   Trial {trial_idx + 1}/{config.num_trials_per_task}")
        
        try:
            # Run episode
            episode_data = run_single_episode(
                controller, env, task_description, 
                initial_states[trial_idx], config
            )
            
            # Save video
            save_episode_video(episode_data, task.name, trial_idx, config)
            
            # Track results
            task_results.append(episode_data)
            if episode_data["success"]:
                successes += 1
            
            logging.info(f"   {'✅' if episode_data['success'] else '❌'} "
                        f"Steps: {episode_data['steps']}, Reward: {episode_data['total_reward']:.3f}")
            
        except Exception as e:
            logging.error(f"   ❌ Trial {trial_idx + 1} failed: {e}")
    
    # Cleanup
    env.close()
    
    # Calculate task success rate
    success_rate = successes / len(task_results) if task_results else 0.0
    
    task_summary = {
        "task_id": task_id,
        "task_name": task.name,
        "task_description": task.language,
        "num_trials": len(task_results),
        "successes": successes,
        "success_rate": success_rate,
        "avg_steps": np.mean([r["steps"] for r in task_results]) if task_results else 0,
        "avg_reward": np.mean([r["total_reward"] for r in task_results]) if task_results else 0,
        "episodes": task_results
    }
    
    logging.info(f"   📊 Task success rate: {success_rate:.1%} ({successes}/{len(task_results)})")
    
    return task_summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Libero Mock Test")
    parser.add_argument("--task-suite", default="libero_90", 
                       choices=["libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal"],
                       help="Task suite to test")
    parser.add_argument("--controller", default="mock",
                       choices=["mock", "random", "openpi"],
                       help="Controller type")
    parser.add_argument("--num-tasks", type=int, default=3,
                       help="Number of tasks to test")
    parser.add_argument("--num-trials", type=int, default=3,
                       help="Number of trials per task")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Max steps per episode")
    parser.add_argument("--no-videos", action="store_true",
                       help="Disable video recording")
    parser.add_argument("--openpi-checkpoint", 
                       default="/mnt/arc/yygx/pkgs_baselines/openpi/checkpoints/pi0_fast_libero_low_mem_finetune_yy/baseline_default/29999",
                       help="OpenPI checkpoint directory")
    parser.add_argument("--openpi-config", default="pi0_fast_libero_low_mem_finetune",
                       help="OpenPI config name")
    
    args = parser.parse_args()
    
    # Create config
    config = TestConfig(
        task_suite_name=args.task_suite,
        num_trials_per_task=args.num_trials,
        num_tasks=args.num_tasks,
        max_steps=args.max_steps,
        controller_type=args.controller,
        save_videos=not args.no_videos,
        openpi_checkpoint_dir=args.openpi_checkpoint,
        openpi_config_name=args.openpi_config
    )
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("🧪 Libero Mock Test")
    print("=" * 40)
    print(f"Task suite: {config.task_suite_name}")
    print(f"Controller: {config.controller_type}")
    print(f"Tasks: {config.num_tasks}")
    print(f"Trials per task: {config.num_trials_per_task}")
    print(f"Max steps: {config.max_steps}")
    print(f"Videos: {'✅' if config.save_videos else '❌'}")
    
    # Create controller
    try:
        controller = create_controller(config)
        print(f"✅ Controller ready: {type(controller).__name__}")
    except Exception as e:
        print(f"❌ Failed to create controller: {e}")
        return 1
    
    # Check if Libero is available - we always need real Libero environment
    if not LIBERO_AVAILABLE:
        print(f"\n❌ Libero is not available! Please install Libero first.")
        print("💡 Install Libero: cd retriever/envs/libero && bash install.sh")
        return 1
    
    # Full Libero evaluation
    print(f"\n🏠 Running full Libero evaluation")
    
    # Load task suite
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[config.task_suite_name]()
        print(f"✅ Loaded task suite: {config.task_suite_name} ({task_suite.n_tasks} tasks)")
    except Exception as e:
        print(f"❌ Failed to load task suite: {e}")
        return 1
    
    # Run evaluation
    print(f"\n🚀 Starting evaluation...")
    
    all_results = []
    total_successes = 0
    total_trials = 0
    
    try:
        for task_id in range(min(config.num_tasks, task_suite.n_tasks)):
            task_result = run_task_evaluation(controller, task_suite, task_id, config)
            all_results.append(task_result)
            
            if "successes" in task_result:
                total_successes += task_result["successes"]
                total_trials += task_result["num_trials"]
        
        # Final summary
        overall_success_rate = total_successes / total_trials if total_trials > 0 else 0.0
        
        print(f"\n🏆 FINAL RESULTS")
        print("=" * 30)
        print(f"Controller: {config.controller_type}")
        print(f"Tasks evaluated: {len(all_results)}")
        print(f"Total trials: {total_trials}")
        print(f"Total successes: {total_successes}")
        print(f"Overall success rate: {overall_success_rate:.1%}")
        
        if config.save_videos:
            print(f"📹 Videos saved to: {config.video_out_path}")
        
        # Per-task summary
        print(f"\n📋 Per-task results:")
        for result in all_results:
            if "success_rate" in result:
                print(f"  {result['task_name']}: {result['success_rate']:.1%} "
                      f"({result['successes']}/{result['num_trials']})")
        
        print(f"\n✅ Libero evaluation complete!")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
