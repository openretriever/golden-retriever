#!/usr/bin/env python3
"""
Controller Flow System

Abstract controller interface and concrete implementations for robot policies.
Supports OpenPI, mock controllers, and custom policy integration.
"""

import sys
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import numpy as np

# Add path for Flow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from retriever.core.flow import Flow
except ImportError:
    print("⚠️  Retriever Flow not available - using basic interface")
    # Fallback base class
    class Flow:
        def __init__(self):
            pass
        
        def run(self, input_data):
            return self(input_data)

try:
    from ..types.robotics_types import RobotObservation, RobotAction
except ImportError:
    from robotics_types import RobotObservation, RobotAction


class ControllerFlow(Flow, ABC):
    """
    Abstract base class for robot controllers in the Flow system.
    
    All robot controllers should inherit from this class and implement:
    - load_policy(): Initialize the policy
    - _get_action(): Get action from observation
    
    This provides a unified interface for:
    - OpenPI policies
    - RT-1 models
    - Custom trained policies  
    - Rule-based controllers
    - Mock/test controllers
    """
    
    def __init__(self):
        super().__init__()
        self._policy = None
        self._is_loaded = False
        self._step_count = 0
    
    @abstractmethod
    def load_policy(self) -> None:
        """Load the policy implementation and set self._is_loaded = True"""
        pass
    
    def run(self, obs: RobotObservation) -> RobotAction:
        """
        Main entry point - convert observation to action.
        
        This handles policy loading and calls the implementation-specific _get_action().
        """
        if not self._is_loaded:
            self.load_policy()
        
        self._step_count += 1
        return self._get_action(obs)
    
    @abstractmethod
    def _get_action(self, obs: RobotObservation) -> RobotAction:
        """Implementation-specific action generation"""
        pass
    
    def __call__(self, obs: RobotObservation) -> RobotAction:
        """Allow direct calling: action = controller(observation)"""
        return self.run(obs)


class MockControllerFlow(ControllerFlow):
    """
    Mock controller for testing and development.
    
    Generates simple deterministic actions that create visible robot movement.
    Perfect for testing the full pipeline without needing a trained policy.
    """
    
    def __init__(self, 
                 n_joints: int = 6,  # Libero uses 6-DOF end-effector control
                 movement_amplitude: float = 0.1,
                 movement_frequency: float = 0.05):
        super().__init__()
        self.n_joints = n_joints
        self.movement_amplitude = movement_amplitude
        self.movement_frequency = movement_frequency
        self._is_loaded = True  # Mock is always "loaded"
    
    def load_policy(self) -> None:
        """Mock policy doesn't need loading"""
        print("🎭 Mock controller ready")
    
    def _get_action(self, obs: RobotObservation) -> RobotAction:
        """Generate simple sinusoidal movements"""
        
        # Create base joint positions (small movements around zero)
        joint_positions = np.zeros(self.n_joints, dtype=np.float32)
        
        # Add sinusoidal movement to create visible motion
        for i in range(self.n_joints):
            phase = self._step_count * self.movement_frequency + i * 0.5
            joint_positions[i] = self.movement_amplitude * np.sin(phase)
        
        # Simple gripper control: slowly open and close
        gripper_phase = self._step_count * self.movement_frequency * 0.3
        gripper_action = 0.5 * np.sin(gripper_phase)
        
        return RobotAction(
            joint_positions=joint_positions,
            gripper_action=gripper_action,
            metadata={
                "controller": "mock",
                "step": self._step_count,
                "movement_type": "sinusoidal"
            }
        )


class OpenPIControllerFlow(ControllerFlow):
    """
    OpenPI policy wrapped as a Flow controller.
    
    Loads and runs the π₀ model directly in-process (no server required).
    """
    
    def __init__(self, 
                 checkpoint_dir: str,
                 config_name: str = "pi0_fast_libero_low_mem_finetune",
                 resize_size: int = 224,
                 replan_steps: int = 5,
                 default_prompt: str = None):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.config_name = config_name
        self.resize_size = resize_size
        self.replan_steps = replan_steps
        self.default_prompt = default_prompt
        self._policy = None
        self._action_plan = []
    
    def load_policy(self) -> None:
        """Load OpenPI policy directly in-process"""
        try:
            # Add OpenPI to path
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/openpi/src'))
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/openpi/packages/openpi-client/src'))
            
            # Import OpenPI modules
            from openpi.policies import policy_config
            from openpi.training import config as _config
            from openpi_client import image_tools
            
            # Store image tools for preprocessing
            self._image_tools = image_tools
            
            # Load config and create policy
            print(f"🤖 Loading OpenPI model from {self.checkpoint_dir}")
            train_config = _config.get_config(self.config_name)
            
            self._policy = policy_config.create_trained_policy(
                train_config=train_config,
                checkpoint_dir=self.checkpoint_dir,
                default_prompt=self.default_prompt
            )
            
            self._is_loaded = True
            print(f"✅ OpenPI policy loaded successfully")
            
        except ImportError as e:
            raise ImportError(f"OpenPI modules not available: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenPI policy: {e}")
    
    def _get_action(self, obs: RobotObservation) -> RobotAction:
        """Get action from OpenPI policy (direct inference)"""
        
        # Check if we need to replan (action buffer empty)
        if len(self._action_plan) == 0:
            # Prepare observation for OpenPI
            openpi_obs = self._prepare_openpi_observation(obs)
            
            # Direct policy inference
            try:
                response = self._policy.infer(openpi_obs)
                action_chunk = response["actions"]
                # Store actions for next few steps
                self._action_plan = list(action_chunk[:self.replan_steps])
            except Exception as e:
                print(f"❌ OpenPI inference failed: {e}")
                return RobotAction(
                    joint_positions=np.zeros(6, dtype=np.float32),
                    gripper_action=0.0,
                    metadata={"controller": "openpi", "error": str(e)}
                )
        
        # Get next action from plan
        if self._action_plan:
            raw_action = self._action_plan.pop(0)
            
            # Convert to RobotAction format
            # OpenPI returns [x, y, z, rx, ry, rz, gripper] for Libero
            if len(raw_action) >= 7:
                joint_positions = raw_action[:6]  # End-effector pose
                gripper_action = raw_action[6]    # Gripper control
            else:
                joint_positions = np.zeros(6, dtype=np.float32)
                gripper_action = 0.0
            
            return RobotAction(
                joint_positions=np.array(joint_positions, dtype=np.float32),
                gripper_action=float(gripper_action),
                metadata={
                    "controller": "openpi",
                    "step": self._step_count,
                    "plan_remaining": len(self._action_plan)
                }
            )
        else:
            # Fallback: return zero action
            return RobotAction(
                joint_positions=np.zeros(6, dtype=np.float32),
                gripper_action=0.0,
                metadata={"controller": "openpi", "error": "no_action_available"}
            )
    
    def _prepare_openpi_observation(self, obs: RobotObservation) -> Dict[str, Any]:
        """Convert RobotObservation to OpenPI format"""
        
        openpi_obs = {}
        
        # Process images
        if "agentview" in obs.images:
            img = obs.images["agentview"]
            # Resize and convert to format expected by OpenPI
            img_resized = self._image_tools.resize_with_pad(img, self.resize_size, self.resize_size)
            img_uint8 = self._image_tools.convert_to_uint8(img_resized)
            openpi_obs["observation/image"] = img_uint8
        
        # Check for wrist camera (Libero uses "wrist" key in our conversion)
        if "wrist" in obs.images:
            img = obs.images["wrist"]
            # Resize and convert to format expected by OpenPI
            img_resized = self._image_tools.resize_with_pad(img, self.resize_size, self.resize_size)
            img_uint8 = self._image_tools.convert_to_uint8(img_resized)
            openpi_obs["observation/wrist_image"] = img_uint8
        
        # Robot state
        openpi_obs["observation/state"] = obs.robot_state
        
        # Task description
        openpi_obs["prompt"] = obs.task_info
        
        return openpi_obs
    



class RandomControllerFlow(ControllerFlow):
    """
    Random controller for testing environment dynamics.
    
    Generates random but bounded actions to test environment response.
    """
    
    def __init__(self, 
                 n_joints: int = 6,
                 action_scale: float = 0.05,
                 seed: Optional[int] = None):
        super().__init__()
        self.n_joints = n_joints
        self.action_scale = action_scale
        self._rng = np.random.RandomState(seed)
        self._is_loaded = True
    
    def load_policy(self) -> None:
        """Random controller doesn't need loading"""
        print("🎲 Random controller ready")
    
    def _get_action(self, obs: RobotObservation) -> RobotAction:
        """Generate random bounded actions"""
        
        # Random joint positions (small movements)
        joint_positions = self._rng.uniform(
            -self.action_scale, 
            self.action_scale, 
            size=self.n_joints
        ).astype(np.float32)
        
        # Random gripper action
        gripper_action = self._rng.uniform(-1.0, 1.0)
        
        return RobotAction(
            joint_positions=joint_positions,
            gripper_action=gripper_action,
            metadata={
                "controller": "random",
                "step": self._step_count,
                "seed": self._rng.get_state()[1][0]  # Current random state
            }
        )


if __name__ == "__main__":
    # Test the controller flows
    print("🤖 Testing Controller Flow System")
    print("=" * 40)
    
    # Create test observation
    test_obs = RobotObservation(
        images={
            "agentview": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            "wrist": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        },
        robot_state=np.random.randn(9).astype(np.float32),
        task_info="pick up the red cup"
    )
    
    # Test MockControllerFlow
    print("\n🎭 Testing MockControllerFlow...")
    mock_controller = MockControllerFlow()
    
    for i in range(3):
        action = mock_controller(test_obs)
        print(f"  Step {i+1}: joints={action.joint_positions[:3]}, gripper={action.gripper_action:.2f}")
    
    # Test RandomControllerFlow
    print("\n🎲 Testing RandomControllerFlow...")
    random_controller = RandomControllerFlow(seed=42)
    
    for i in range(3):
        action = random_controller(test_obs)
        print(f"  Step {i+1}: joints={action.joint_positions[:3]}, gripper={action.gripper_action:.2f}")
    
    # Test OpenPIControllerFlow (will fail without server)
    print("\n🤖 Testing OpenPIControllerFlow...")
    try:
        openpi_controller = OpenPIControllerFlow()
        # This will fail without a running server, which is expected
        action = openpi_controller(test_obs)
        print(f"  ✅ OpenPI action: joints={action.joint_positions[:3]}, gripper={action.gripper_action:.2f}")
    except Exception as e:
        print(f"  ⚠️  OpenPI test failed (expected without server): {e}")
    
    print("\n✅ Controller Flow system ready!")
    print("💡 Use MockControllerFlow for testing without trained policies")
    print("🔗 Use OpenPIControllerFlow with a running policy server")
