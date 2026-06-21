
import argparse
import sys
import numpy as np
import time
import logging
from dataclasses import dataclass

try:
    import torch
    import transformers
except ImportError:
    print("PyTorch and Transformers required. Run with `pixi run -e torch`.")
    sys.exit(0)

from retriever.flow import Flow, io, Rate
from retriever import connect, Pipeline, run
from retriever.flow.adapter import Latest

from examples.advanced.hierarchical_vla.perception import PerceptionFlow, GoalEmbedding
from examples.advanced.hierarchical_vla.control import ControlFlow, ControlInput, RobotState, RobotAction
from retriever.lib.hf import TransformerInput

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HierarchicalVLA")

# Mock State Provider (Simulates Robot Hardware Driver)
@io
@dataclass
class Nothing:
    pass

class RobotDriver(Flow[Nothing, RobotState]):
    def run(self, input: Nothing) -> RobotState:
        # Generate random state (5 joints)
        return RobotState(
            joint_angles=np.random.randn(5).astype(np.float32),
            timestamp=time.time()
        )

# Mock Command Provider (Simulates User Input)
class CommandGenerator(Flow[Nothing, TransformerInput]):
    def run(self, input: Nothing) -> TransformerInput:
        return TransformerInput(text="Pick up the red block")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="dora", choices=["dora", "multiprocessing"])
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    # Define frequencies
    FAST_HZ = 50
    SLOW_HZ = 1  # 50x slower

    # 1. Define Flows
    driver = RobotDriver() @ Rate(hz=FAST_HZ)
    cmd_gen = CommandGenerator() @ Rate(hz=SLOW_HZ) # Commands update slowly
    
    perception = PerceptionFlow() @ Rate(hz=SLOW_HZ) # Runs when cmd arrives
    control = ControlFlow() @ Rate(hz=FAST_HZ)       # Runs when driver updates (fast)

    # 2. Build Pipeline
    # Dataflow:
    # driver [50Hz] -> robot_state
    # cmd_gen [1Hz] -> text -> perception -> goal
    
    # We want Control to run at 50Hz.
    # It needs (State, Goal).
    # State arrives at 50Hz. Goal arrives at 1Hz.
    # We must sample the Goal "Latest" to match the State's rate.

    # Connections
    # 1. Command -> Perception
    connect(cmd_gen, perception)
    
    # 2. Merging inputs for Control
    # Driver outputs: joint_angles, timestamp. Map to ControlInput fields.
    connect(driver, control, map={
        "joint_angles": "joint_angles", 
        "timestamp": "state_timestamp"
    })
    
    # Perception outputs: vector, timestamp. Map to ControlInput fields.
    connect(perception, control, map={
        "vector": "goal_vector", 
        "timestamp": "goal_timestamp"
    }, sync=Latest())

    # 3. Run
    logger.info(f"Starting Hierarchical VLA Demo (Backend: {args.backend})")
    logger.info(f"Perception: ~{SLOW_HZ} Hz | Control: ~{FAST_HZ} Hz")
    
    run(
        actions=None, # Run default pipeline
        backend=args.backend,
        duration=args.duration,
        log_level="INFO"
    )

if __name__ == "__main__":
    main()
