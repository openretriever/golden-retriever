"""
Retriever Advanced Example: VLA Inference Optimization (Manual Buffer Flow)

This demo showcases the "Manual Buffer" approach where a dedicated Flow node (`ActionBuffer`)
manages the queue of action chunks from the VLA.

It demonstrates:
1.  OpenPIFlow Pattern: Standardized dataflow for VLA inference.
2.  Action Buffering: Explicit buffering logic in a Flow node.
3.  Dora Backend: Running VLA components in separate dataflow nodes.

Usage:
    pixi run python -m examples.advanced.vla_inference_optim.demo_buffer_flow
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from datetime import datetime

# Ensure project root is in path (Must happen before imports)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
src_root = os.path.join(project_root, "src")
sys.path.insert(0, src_root)
# isort: off
from retriever.flow import Pipeline, Rate, Flow, io, Latest
import retriever
try:
    from .mock_vla_node import MockVLAFlow, VLAInput, VLAAction
except ImportError:
    from mock_vla_node import MockVLAFlow, VLAInput, VLAAction

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

@io
@dataclass
class SingleAction:
    action: np.ndarray
    timestamp: float

class ActionBuffer(Flow[VLAAction, SingleAction]):
    """
    Buffers action chunks from VLA and streams single actions to the robot.
    Handles 'underrun' (no actions) and 'overrun' (merging/overwriting).
    """
    def init(self):
        self.queue = [] # Simple list as queue (of actions)
        self.last_vla_timestamp = 0.0
        self._last_log_time = time.time()
        
    def run(self, vla_input: VLAAction) -> Optional[SingleAction]:
        # 1. Ingest new chunk
        if vla_input:
            # Check for None timestamp issue
            if vla_input.timestamp is None:
                print(f"[Buffer] WARNING: Received VLAAction with None timestamp! Input: {vla_input}")
                # return None # Don't return here, might still have queue

            # Only accept newer chunks
            input_ts = vla_input.timestamp if vla_input.timestamp is not None else 0.0
            
            if input_ts > self.last_vla_timestamp:
                self.last_vla_timestamp = input_ts
                # Extract actions list
                new_chunk = vla_input.action.tolist() if isinstance(vla_input.action, np.ndarray) else vla_input.action
                self.queue = new_chunk
                # print(f"[Buffer] Refill. Latency: {vla_input.latency*1000:.1f}ms. Steps: {len(new_chunk)}")

        # Log buffer depth for visualization
        if (time.time() - self._last_log_time > 0.5):
             # print(f"[Buffer] Status: Depth={len(self.queue)} ts={datetime.fromtimestamp(time.time()).strftime('%H:%M:%S.%f')[:-3]}")
             self._last_log_time = time.time()

        if not self.queue:
            # Underrun handling
            return None
            
        # Temporal Alignment Strategy (Fast-Forward)
        # We have a chunk starting at T_obs. We are at T_now.
        # Which index 'k' should we execute?
        
        # 1. Get current time
        now = time.time()
        
        # 2. Latest chunk info 
        chunk_start_time = self.last_vla_timestamp
        # If timestamp was None, fallback?
        if chunk_start_time == 0.0:
             chunk_start_time = now # Just play immediately if no timestamp
             
        dt = 0.1 # Fixed for now/mock
        
        # 3. Calculate elapsed time from the START of the chunk observation
        delta_t = now - chunk_start_time
        
        # 4. Calculate Index
        # k = (now - t_start) / dt
        k = int(round(delta_t / dt))
        
        # 5. Validity Check
        if k < 0:
            return None
        elif k >= len(self.queue):
             # Overrun / Expired Chunk
             return None
        
        # 6. Execute Specific Index
        action_step = self.queue[k]
        
        return SingleAction(
            action=np.array(action_step),
            timestamp=now
        )

class VLAValidationSink(Flow[SingleAction, None]):
    """Validates the alignment of VLA actions and Visualizes Trajectories."""
    def init(self):
        self.history = [] # (ts, action_vec)
        self.start_time = time.time()

    def save_plot(self):
        # ... (Same as before, simplified for no plot saving in execution check)
        pass

    def run(self, action: SingleAction):
        if action is None or action.action is None:
            return

        # Store for vis
        if action.timestamp is not None:
            self.history.append((float(action.timestamp), action.action))
            
        # Log periodically
        if len(self.history) % 50 == 0:
             print(f"[Sink] Executed {len(self.history)} steps. Last: {action.action[0]:.3f}")

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
    parser.add_argument("--duration", type=float, default=120.0, help="Run duration in seconds.")
    args = parser.parse_args()

    # Set global default sync policy
    retriever.init(default_sync=Latest())

    pipe = Pipeline("openpi_vla_demo_manual")

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

        # 4. Action Buffer Node (Manual)
        buffer = ActionBuffer() @ Rate(hz=50)
        pipe.connect(vla, buffer, sync=Latest())

        # 5. Sink
        sink = VLAValidationSink() @ Rate(hz=50)
        pipe.connect(buffer, sink, sync=Latest())

    # Validate to IR for the terminal summary, then use the public Pipeline
    # visualization API for the interactive HTML artifact.
    try:
        from retriever.ir.viz import generate_ascii_graph
        print("\n" + "="*60)
        print("Pipeline Structure:")
        print("="*60)
        ir = pipe.validate()
        print(generate_ascii_graph(ir))
        print("="*60 + "\n")
        html_path = pipe.visualize("vla_pipeline_manual.html")
        print(f"Interactive graph: {html_path}")
    except ImportError:
        print("Visualization module not found.")

    try:
        # Run for 2 minutes
        pipe.run(backend=args.backend, duration=args.duration)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
