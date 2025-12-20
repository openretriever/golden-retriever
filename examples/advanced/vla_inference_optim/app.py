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
    pixi run demo-vla
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
from retriever.flow import Pipeline, Rate, Flow, flow_io, Latest
# from retriever.flow.vla.openpi import OpenPIFlow, VLAInput, VLAAction
try:
    from .mock_vla_node import MockVLAFlow, VLAInput, VLAAction
except ImportError:
    from mock_vla_node import MockVLAFlow, VLAInput, VLAAction


@flow_io
@dataclass
class RobotState:
    joint_positions: list[float]
    joint_velocities: list[float]
    timestamp: float

@flow_io
@dataclass
class CameraImage:
    data: str

@flow_io
@dataclass
class VLAAdapterInput:
    state: RobotState
    img: CameraImage

@flow_io
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

        # print(f"[RobotSource] Emitting state. TS: {now:.3f}")
        return RobotOutput(state=RobotState(
            joint_positions=self.q.tolist(),
            joint_velocities=self.dq.tolist(),
            timestamp=now
        ))

@flow_io
@dataclass
class CameraOutput:
    img: CameraImage

class DummyCameraSource(Flow[None, CameraOutput]):
    """Simulates a camera source."""
    def run(self, _):
        return CameraOutput(img=CameraImage(data="dummy_image_tensor"))

@flow_io
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
        self.queue = [] # Simple list as queue
        self.last_vla_timestamp = 0.0
        
    def run(self, vla_input: VLAAction) -> Optional[SingleAction]:
        # 1. Ingest new chunk
        if vla_input:
            # Check for None timestamp issue
            if vla_input.timestamp is None:
                print(f"[Buffer] WARNING: Received VLAAction with None timestamp! Input: {vla_input}")
                return None

            if vla_input.timestamp > self.last_vla_timestamp:
                self.last_vla_timestamp = vla_input.timestamp
                new_chunk = vla_input.action.tolist()
                self.queue = new_chunk
                print(f"[Buffer] Refill. Latency: {vla_input.latency*1000:.1f}ms. Steps: {len(new_chunk)}")

        # 2. Dequeue
        if self.queue:
             # Check if we have enough items, or just pop
             pass

        # Log buffer depth for visualization
        if hasattr(self, '_last_log_time') and (time.time() - self._last_log_time > 0.5):
             print(f"[Buffer] Status: Depth={len(self.queue)} ts={time.time()}")
             self._last_log_time = time.time()
        elif not hasattr(self, '_last_log_time'):
             self._last_log_time = time.time()

        if not self.queue:
            # Underrun handling
            # print("[Buffer] Underrun: No actions available.")
            return None
            
        # Temporal Alignment Strategy (Fast-Forward)
        # We have a chunk starting at T_obs. We are at T_now.
        # Which index 'k' should we execute?
        
        # 1. Get current time
        now = time.time()
        
        # 2. Latest chunk info (we only keep the freshest chunk in self.queue for now)
        # Note: self.queue is currently just a list of actions. 
        # We need to store metadata (timestamp, dt) alongside the queue or in the class state.
        # Let's rely on self.last_vla_timestamp and assume dt=0.1 (until we update VLAAction fully or pass it)
        chunk_start_time = self.last_vla_timestamp
        dt = 0.1 # Fixed for now, or get from VLAAction if we stored it
        
        # 3. Calculate elapsed time from the START of the chunk observation
        delta_t = now - chunk_start_time
        
        # 4. Calculate Index
        # k = (now - t_start) / dt
        k = int(round(delta_t / dt))
        
        # 5. Validity Check
        if k < 0:
            # This should technically not happen if now > chunk_start_time (which is in the past)
            # But could happen if clock skew or tight loop?
            return None
        elif k >= len(self.queue):
             # Overrun / Expired Chunk
             # We ran out of actions in this chunk before a new one arrived.
             # print(f"[Buffer] Chunk Expired. k={k}, len={len(self.queue)}")
             return None
        
        # 6. Execute Specific Index
        action_step = self.queue[k]
        
        # print(f"[Buffer] Executing Index k={k}/{len(self.queue)} (Delta={delta_t:.3f}s)")
        
        return SingleAction(
            action=np.array(action_step),
            timestamp=now
        )

class VLAValidationSink(Flow[SingleAction, None]):
    """Validates the alignment of VLA actions and Visualizes Trajectories."""
    def init(self):
        self.history = [] # (ts, action_vec)
        self.start_time = time.time()
        logger.info("[Sink] Visualization initialized. Plot will be saved periodically.")

    def save_plot(self):
        if not self.history:
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg') # Headless mode
            import matplotlib.pyplot as plt
            
            print(f"[Sink] Saving visualization with {len(self.history)} points...")
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
            print(f"[Sink] Plot saved to {filename} (CWD: {os.getcwd()})")
            # self.plot_saved = True # Periodic
        except Exception as e:
            print(f"[Sink] Failed to save plot: {e}")

    def run(self, action: SingleAction):
        if action:
            ts_str = f"{action.timestamp:.3f}" if action.timestamp is not None else "None"
            # print(f"[Sink] Action TS: {ts_str}. Val: {action.action[0]:.4f}...")
            
            # Store for vis
            if action.timestamp is not None:
                self.history.append((float(action.timestamp), action.action))
                
            # Log periodically
            if len(self.history) % 50 == 0:
                 print(f"[Sink] Executed {len(self.history)} steps. Last: {action.action[0]:.3f}")
                 self.save_plot()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="dora", choices=["dora", "multiprocessing"])
    args = parser.parse_args()

    pipe = Pipeline("openpi_vla_demo")

class InputAdapter(Flow[VLAAdapterInput, VLAInput]):
    def run(self, inputs: VLAAdapterInput):
        if not inputs.state or not inputs.img: return None
        
        # print(f"[Adapter] State: {inputs.state}, Timestamp: {inputs.state.timestamp}")
        
        return VLAInput(
            instruction="Pick up the apple", # Static instruction for demo
            image=inputs.img.data,
            state={
                "joint_positions": inputs.state.joint_positions,
                "joint_velocities": inputs.state.joint_velocities
            },
            timestamp=inputs.state.timestamp
        )

# ... (Sink needs to be updated too, using MultiReplace or separate call)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="dora", choices=["dora", "multiprocessing"])
    args = parser.parse_args()

    pipe = Pipeline("openpi_vla_demo")

    with pipe:
        # 1. Sources
        robot = DummyRobotSource() @ Rate(hz=50)
        camera = DummyCameraSource() @ Rate(hz=10)
        
        # 2. Adapter
        adapter = InputAdapter() @ Rate(hz=50)

        # Connect sources to adapter
        pipe.connect(robot, adapter, map={"state": "state"})
        pipe.connect(camera, adapter, map={"img": "img"}) 

        # 3. VLA Node (Mock)
        # Adaptive Rate: Runs as fast as possible (bounded by mock inference time), capped at 100Hz.
        # syncing with 'Latest' input ensures we don't build up lag.
        vla = MockVLAFlow(model_id="mock-vla") @ Rate(hz=100)
        
        # Connect adapter output to VLA input
        # Note: map is implicit if fields match, but explicit map is safer if needed.
        # InputAdapter outputs VLAInput (fields: instruction, image, state, timestamp)
        # MockVLAFlow takes VLAInput.
        pipe.connect(adapter, vla, sync=Latest())

        # 4. Action Buffer
        # Runs at 50Hz to feed the robot smoothly
        buffer = ActionBuffer() @ Rate(hz=50)
        # Connect VLA to Buffer. 
        # Note: We use Latest() so the buffer always sees the freshest chunk available 
        # if it runs faster than VLA. But Buffer logic handles "new timestamp" check.
        pipe.connect(vla, buffer, sync=Latest())

        # 5. Sink
        sink = VLAValidationSink() @ Rate(hz=50)
        pipe.connect(buffer, sink)

    try:
        # Run for 2 minutes to show stability
        pipe.run(backend=args.backend, duration=120.)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
