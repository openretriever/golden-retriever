"""
Multi-Language Robot IK Solver Example

Demonstrates a native Node (IK Solver) implemented in Python, Rust, and C++.
"""

import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from retriever import Flow, Pipeline, Rate, Latest
from retriever.flow.io import flow_io

# ==============================================================================
# Flow I/O Types
# ==============================================================================

@flow_io
@dataclass
class TargetPose:
    # [x, y, z, roll, pitch, yaw]
    pose: np.ndarray 

@flow_io
@dataclass
class JointAngles:
    # [j1, j2, j3, j4, j5, j6]
    joints: np.ndarray

# ==============================================================================
# Flows
# ==============================================================================

class TrajectoryGenerator(Flow[None, TargetPose]):
    """Generates a circular trajectory for the end-effector."""

    def __init__(self):
        self.tick = 0

    def step(self, _) -> TargetPose:
        t = self.tick * 0.1
        self.tick += 1
        
        # Draw a circle in XY plane
        pose = np.array([
            0.5 + 0.1 * np.cos(t),  # x
            0.0 + 0.1 * np.sin(t),  # y
            0.4,                    # z
            0.0, 3.14, 0.0          # r, p, y
        ], dtype=np.float32)
        
        return TargetPose(pose=pose)


class IKSolver(Flow[TargetPose, JointAngles]):
    """
    Inverse Kinematics Solver.
    
    This Python implementation serves as:
    1. A reference implementation
    2. A placeholder for native overrides (Rust/C++)
    """
    
    def step(self, input: TargetPose) -> JointAngles:
        if input.pose is None:
            return JointAngles(joints=np.zeros(6, dtype=np.float32))
            
        # Mock analytical IK for demo purposes
        # In reality, this would be `kdl.cartToJnt(pose)` or similar
        x, y, z = input.pose[0], input.pose[1], input.pose[2]
        
        # Simple mapping just to show data flow
        j1 = np.arctan2(y, x)
        j2 = z * 2.0
        j3 = x + y
        
        joints = np.array([j1, j2, j3, 0.0, 0.0, 0.0], dtype=np.float32)
        return JointAngles(joints=joints)


class RobotDriver(Flow[JointAngles, None]):
    """Receives joint angles and 'drives' the robot."""

    def __init__(self):
        self.count = 0
        self.start_time = 0.0

    def init(self):
        import time
        self.start_time = time.time()

    def step(self, input: JointAngles) -> None:
        if input.joints is not None:
            self.count += 1
            if self.count % 100 == 0:
                print(f"[RobotDriver] Processed {self.count} targets...", flush=True)

    def finalize(self):
        import time
        duration = time.time() - self.start_time
        rate = self.count / duration if duration > 0 else 0.0
        print(f"\n[RobotDriver] STATS: Total={self.count}, Duration={duration:.2f}s, Rate={rate:.2f} Hz")


# ==============================================================================
# Pipeline Definition
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-language IK Solver")
    parser.add_argument("--backend", choices=["python", "rust", "cpp"], default="python")
    parser.add_argument("--rate", type=float, default=50.0, help="Pipeline frequency in Hz")
    parser.add_argument("--duration", type=float, default=None, help="Run duration in seconds")
    args = parser.parse_args()

    pipe = Pipeline("ik_solver_demo")

    gen = TrajectoryGenerator() @ Rate(hz=args.rate)
    ik = IKSolver() @ Rate(hz=args.rate) # Match generator rate
    driver = RobotDriver() @ Rate(hz=args.rate)

    # Explicitly connect flows within the pipeline
    # We use 'map' to define input/output mapping and 'sync' for synchronization policy
    pipe.connect(gen, ik, map={"pose": "pose"}, sync=Latest())
    pipe.connect(ik, driver, map={"joints": "joints"}, sync=Latest())

    # Path to compiled native binaries (relative to this script)
    base_dir = Path(__file__).parent.resolve()
    native_overrides = {}
    
    if args.backend == "rust":
        binary = base_dir / "target/release/rust-controller"
        if not binary.exists():
            raise FileNotFoundError(f"Rust binary not found at {binary}. Build it with `cargo build --release` in `examples/advanced/native_controller`.")
        native_overrides["IKSolver"] = str(binary)
        
    elif args.backend == "cpp":
        binary = base_dir / "build/cpp-controller"
        if not binary.exists():
            raise FileNotFoundError(f"C++ binary not found at {binary}. Build it with `cmake -S examples/advanced/native_controller -B examples/advanced/native_controller/build && cmake --build examples/advanced/native_controller/build --config Release`.")
        native_overrides["IKSolver"] = str(binary)

    print(f"\nStopped pipeline? Press Ctrl+C")
    print(f"Running with {args.backend.upper()} backend at {args.rate} Hz...\n")

    pipe.run(
        backend="multiprocessing",
        native_overrides=native_overrides if native_overrides else None,
        duration=args.duration,
    )

if __name__ == "__main__":
    main()
