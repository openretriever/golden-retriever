"""
Example: Using OpenPI Flow Modules in Retriever

This script demonstrates how to use both the local and remote OpenPI flow modules
in Retriever's Flow system, including composition into a pipeline.

The example observation includes dummy image data and robot state fields
that match typical OpenPI input formats (e.g., for DROID or ALOHA robots).
"""

import numpy as np
from retriever.core.flow import Flow

# Import the OpenPI flow module creators
from examples.openpi.openpi_local_flow_module import make_openpi_local_policy
from examples.openpi.openpi_remote_flow_module import make_openpi_remote_client

def example_observation():
    # Example: DROID-style observation with images and state
    obs = {
        "observation/exterior_image_1_left": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "pick up the fork on the table"
    }
    return obs


def run_local_openpi_flow():
    print("\n--- Local OpenPI Flow Example ---")
    openpi_policy = make_openpi_local_policy()
    openpi_flow = Flow.from_module(openpi_policy)
    obs = example_observation()
    print("Input observation (truncated):", {k: (v.shape if hasattr(v, 'shape') else v) for k, v in obs.items()})
    action = openpi_flow(obs)
    print("Local OpenPI Action:", action)


def run_remote_openpi_flow():
    print("\n--- Remote OpenPI Flow Example ---")
    openpi_client = make_openpi_remote_client(host="localhost", port=8000)
    openpi_flow = Flow.from_module(openpi_client)
    obs = example_observation()
    print("Input observation (truncated):", {k: (v.shape if hasattr(v, 'shape') else v) for k, v in obs.items()})
    action = openpi_flow(obs)
    print("Remote OpenPI Action:", action)


def run_composed_pipeline():
    print("\n--- Composed Pipeline Example ---")
    # Example: Compose a dummy perception flow with OpenPI
    def dummy_perception(obs):
        # Simulate a perception module that adds a processed image
        obs = dict(obs)
        obs["perception/processed_image"] = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        return obs
    perception_flow = Flow.from_module(dummy_perception)
    openpi_policy = make_openpi_local_policy()
    openpi_flow = Flow.from_module(openpi_policy)
    pipeline = perception_flow.then(openpi_flow)
    obs = example_observation()
    print("Input observation (truncated):", {k: (v.shape if hasattr(v, 'shape') else v) for k, v in obs.items()})
    action = pipeline(obs)
    print("Composed Pipeline Action:", action)


if __name__ == "__main__":
    run_local_openpi_flow()
    run_remote_openpi_flow()
    run_composed_pipeline() 