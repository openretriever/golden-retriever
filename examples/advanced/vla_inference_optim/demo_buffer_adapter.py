"""
Retriever Advanced Example: VLA Inference Optimization (Mock VLA / OpenPI Architecture)

This demo showcases a "Real" VLA pipeline architecture designed for the OpenPI library (pi0.5 model),
but uses a MOCK VLA node to simulate realistic inference conditions (Latency, Jitter).

It demonstrates:
1.  OpenPIFlow Pattern: Standardized dataflow for VLA inference.
2.  Adaptive Rate Control: Optimizing for 10Hz inference with 100-200ms latency.
3.  Action Buffering: Stabilizing the output stream to the robot (50Hz) despite VLA jitter.
4.  Dora Backend: Running VLA components in separate dataflow nodes.

Usage:
    python -m examples.advanced.vla_inference_optim.app
    # Or directly as a module:
    pixi run python -m examples.advanced.vla_inference_optim.app
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Ensure project root is in path (Must happen before imports)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
src_root = os.path.join(project_root, "src")
sys.path.insert(0, src_root)
# isort: off
from retriever.flow import Pipeline, Rate, Flow, io, Latest
import retriever
# from retriever.flow.vla.openpi import OpenPIFlow, VLAInput, VLAAction
# ... imports
try:
    from .mock_vla_node import MockVLAFlow, VLAInput, VLAAction
    from .custom_adapters import ActionChunking, SingleAction
except ImportError:
    from mock_vla_node import MockVLAFlow, VLAInput, VLAAction
    from custom_adapters import ActionChunking, SingleAction


@io
@dataclass
class RobotState:
    joint_positions: list[float]
    joint_velocities: list[float]
    timestamp: float

@io
@dataclass
class CameraImage:
    data: str

@io
@dataclass
class VLAAdapterInput:
    state: RobotState
    img: CameraImage

@io
@dataclass
class RobotOutput:
    state: RobotState

class DummyRobotSource(Flow[None, RobotOutput]):
    """Simulates a robot publishing state at high frequency (50Hz)."""
    def init(self):
        self.q = np.zeros(7)
        self.dq = np.random.uniform(-0.1, 0.1, 7) # Constant velocity motion
        self.start_time = time.time()

    def run(self, _):
        now = time.time()
        # Simulate motion
        self.q += self.dq * 0.02 # 50Hz step approx
        # Bounce bounds
        for i in range(7):
            if self.q[i] > 1.0 or self.q[i] < -1.0:
                self.dq[i] *= -1
        return RobotOutput(state=RobotState(
            joint_positions=self.q.tolist(),
            joint_velocities=self.dq.tolist(),
            timestamp=now
        ))

@io
@dataclass
class CameraOutput:
    img: CameraImage

class DummyCameraSource(Flow[None, CameraOutput]):
    """Simulates a camera source."""
    def run(self, _):
        return CameraOutput(img=CameraImage(data="dummy_image_tensor"))

# (ActionBuffer removed - replaced by Adapter)

class VLAValidationSink(Flow[SingleAction, None]):
    """Validates the alignment of VLA actions and Visualizes Trajectories."""
    def init(self):
        self.history = [] # (ts, action_vec)
        self.start_time = time.time()
        # logger.info("[Sink] Visualization initialized.")

    def save_plot(self):
        if not self.history:
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg') # Headless mode
            import matplotlib.pyplot as plt
            
            # print(f"[Sink] Saving visualization with {len(self.history)} points...")
            data = np.array(self.history, dtype=object)
            times = np.array([x[0] - self.start_time for x in data])
            actions = np.vstack([x[1] for x in data]) # (N, 7)
            
            plt.figure(figsize=(10, 6))
            
            # Plot first 3 joints
            for i in range(min(3, actions.shape[1])):
                plt.plot(times, actions[:, i], label=f'Joint {i+1}')
                
            plt.xlabel('Time (s)')
            plt.ylabel('Action Value')
            plt.title('VLA Executed Trajectory (Smoothness Check)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            filename = "vla_trajectory_latest.png"
            plt.savefig(filename)
            plt.close()  # Prevent memory leak from accumulating figures
            # print(f"[Sink] Plot saved to {filename}")
        except Exception as e:
            print(f"[Sink] Failed to save plot: {e}")

    def run(self, action: SingleAction):
        if action.action is None:
            return

        # Store for vis
        if action.timestamp is not None:
            self.history.append((float(action.timestamp), action.action))
            
        # Log periodically
        if len(self.history) % 50 == 0:
             print(f"[Sink] Executed {len(self.history)} steps. Last: {action.action[0]:.3f}")
             self.save_plot()

class InputAdapter(Flow[VLAAdapterInput, VLAInput]):
    def run(self, inputs: VLAAdapterInput):
        if not inputs.state or not inputs.img: return None
        return VLAInput(
            instruction="Pick up the apple",
            image=inputs.img.data,
            state={
                "joint_positions": inputs.state.joint_positions,
                "joint_velocities": inputs.state.joint_velocities
            },
            timestamp=inputs.state.timestamp
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="dora", choices=["dora", "multiprocessing"])
    args = parser.parse_args()

    # Set global default sync policy
    retriever.init(default_sync=Latest())

    pipe = Pipeline("openpi_vla_demo")

    with pipe:
        # 1. Sources
        robot = DummyRobotSource() @ Rate(hz=50)
        camera = DummyCameraSource() @ Rate(hz=10)
        
        # 2. Adapter Node (Data Prep)
        adapter = InputAdapter() @ Rate(hz=50)
        pipe.connect(robot, adapter, map={"state": "state"})
        pipe.connect(camera, adapter, map={"img": "img"}) 

        # 3. VLA Node (Mock)
        vla = MockVLAFlow(model_id="mock-vla") @ Rate(hz=100)
        pipe.connect(adapter, vla, sync=Latest())

        # 4. Sink (Simulates Robot Controller)
        # We replace the manual ActionBuffer with a direct connection
        # using our custom ActionChunk adapter.
        sink = VLAValidationSink() @ Rate(hz=50)
        
        # Combine connections with per-port sync policy
        pipe.connect(
            vla, sink, 
            map={"action": "action", "timestamp": "timestamp"}, 
            sync={
                "action": ActionChunking(dt=0.1),
                "timestamp": Latest()
            }
        )

    # Visualization
    try:
        from retriever.ir.viz import generate_ascii_graph, save_interactive_html
        print("\n" + "="*60)
        print("Pipeline Structure:")
        print("="*60)
        ir = pipe.build_ir()
        print(generate_ascii_graph(ir))
        print("="*60 + "\n")
        save_interactive_html(ir, "vla_pipeline_adapter.html")
    except ImportError:
        print("Visualization module not found.")

    try:
        # Run for 2 minutes
        pipe.run(backend=args.backend, duration=120.)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
