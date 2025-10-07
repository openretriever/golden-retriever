#!/usr/bin/env python3
"""
Integration Demo: Using OpenPI Controller with Libero Environment
"""

import numpy as np
from openpi_controller.flows import OpenPIControllerFlow, MockControllerFlow
from openpi_controller.types import RobotObservation, RobotAction

def main():
    """Run a simple integration demo."""
    print("🚀 Starting Integration Demo...")
    
    # Initialize controller
    controller = MockControllerFlow()
    print(f"✅ Controller initialized: {type(controller).__name__}")
    
    # Create a mock observation
    mock_obs = RobotObservation(
        images={"agentview": np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)},
        robot_state=np.random.randn(7).astype(np.float32),
        task_info="Test task"
    )
    print("✅ Mock observation created")
    
    # Get action from controller
    action = controller.run(mock_obs)
    print(f"✅ Action generated: {type(action).__name__}")
    
    # Test format converters
    from openpi_controller.converters import quat_to_axis_angle, axis_angle_to_quat
    
    # Test quaternion conversion
    quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    axis_angle = quat_to_axis_angle(quat)
    back_to_quat = axis_angle_to_quat(axis_angle)
    print(f"✅ Quaternion conversion test: {np.allclose(quat, back_to_quat)}")
    
    print("🎉 Demo completed successfully!")
    print("\nTo use with Libero environment:")
    print("1. Install Libero dependencies: pip install -e .[libero]")
    print("2. Import: from libero.envs import OffScreenRenderEnv")

if __name__ == "__main__":
    main()
