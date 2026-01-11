# Run: pixi run demo-manipulation-chunking
#
# ============================================================================
# Manipulation Pipeline with Action Chunking
# ============================================================================
#
# This demo extends the standard manipulation pipeline with **Action Chunking**
# for VLA-style control. It reuses the existing flows from the manipulation
# pipeline and adds:
#
# 1. **VLAChunkingPolicy** replaces SkillPolicy - outputs 10-step action chunks
# 2. **ActionBufferFlow** - explicit temporal alignment (visible in graph)
# 3. **Hybrid Clocks** - VLA and TaskPlanner use Rate + Trigger
#
# Architecture:
#
#   CameraSource (30Hz)
#        │
#        ▼
#   PerceptionFlow ────────────────────────────────┐
#        │                                         │
#        ▼ state, atoms                           ▼
#   BeliefUpdaterFlow ◄───────────────────── TaskPlannerFlow
#        │       ▲                             @ Hybrid(1Hz + Trigger)
#        │       │ action                           │
#        │       │                                 │ plan
#        ▼ belief│                                 ▼
#   ╔═══════════════════════════════════════════════════════════╗
#   ║  VLAChunkingPolicy @ Hybrid(2.5Hz + replan_request)       ║
#   ╚═══════════════════════════════════════════════════════════╝
#        │                           ▲
#        │ action_chunk             │ replan_request
#        ▼                          │
#   ╔════════════════════════════════════╗
#   ║  ActionBufferFlow @ 50Hz           ║
#   ║  (temporal alignment: k=δt/dt)     ║
#   ╚════════════════════════════════════╝
#        │
#        │ single action
#        ▼
#   RobotControllerSink @ 200Hz
#        │
#        ▼ status
#   ExecutionMonitorFlow ──────────────────► TaskPlannerFlow (replan)
#
# Key Rates:
# - Camera:       30 Hz
# - TaskPlanner:  Hybrid(1Hz + Trigger on state)
# - VLA Policy:   Hybrid(2.5Hz + Trigger on replan_request)
# - ActionBuffer: 50 Hz
# - Controller:   200 Hz
#
# ============================================================================

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

import retriever
from retriever.flow import Flow, Hybrid, Latest, Pipeline, Rate, Trigger, io

# --- Reuse existing flows from closed_loop_planning ---
from ..flows.belief_updater import BeliefUpdaterFlow
from ..flows.monitor_execution import ExecutionMonitorFlow
from ..flows.perception import PerceptionFlow
from ..flows.planner_astar import TaskPlannerFlow
from ..types.belief import BeliefState
from retriever.types.options import Option


# ============================================================================
# Data Types
# ============================================================================

@io
class CameraOutput:
    """Raw sensor output from camera."""
    data: dict  # Matches PerceptionFlow input


@io
class ActionChunk:
    """
    Time-indexed trajectory of actions from VLA policy.
    H steps at dt intervals = 1 second horizon.
    """
    action_chunk: List[np.ndarray] = field(default_factory=list)
    timestamp: float = 0.0
    dt: float = 0.1
    status: str = "running"


@io
class SingleAction:
    """Single action from ActionBuffer → Controller."""
    action: Optional[np.ndarray] = None
    chunk_index: int = 0
    status: str = ""


@io
class ReplanRequest:
    """Request from ActionBuffer → VLA to regenerate chunk."""
    should_replan: bool = False
    reason: str = ""


@io
class ControlCommand:
    """Command sent to robot actuators."""
    action: Optional[np.ndarray] = None


# ============================================================================
# Flow Nodes
# ============================================================================

class CameraSourceFlow(Flow[None, CameraOutput]):
    """Camera sensor at 30Hz. Matches PerceptionFlow input."""
    def __init__(self, name: str = "CameraSourceFlow"):
        self.name = name
        self._frame_count = 0

    def step(self, inp: None) -> CameraOutput:
        self._frame_count += 1
        return CameraOutput(data={
            "rgb": f"frame_{self._frame_count}",
            "depth": None,
            "timestamp": time.time()
        })


# --- VLA Policy Input/Output Types (compatible with existing flows) ---

@io
class VLAPolicyInput:
    """Input to VLA policy - matches BeliefUpdaterFlow + TaskPlannerFlow outputs."""
    state: BeliefState = None  # type: ignore[assignment] # BeliefState from belief
    plan: List[Option] = field(default_factory=list)  # Plan from planner
    replan_request: bool = False        # From ActionBuffer


class VLAChunkingPolicyFlow(Flow[VLAPolicyInput, ActionChunk]):
    """
    VLA policy that outputs action chunks.

    Runs on Hybrid(2.5Hz + replan_request trigger):
    - Base rate: 2.5Hz (400ms inference time)
    - Can be triggered early by ActionBuffer if chunk expires
    """
    def __init__(self, name: str = "VLAChunkingPolicy", horizon: int = 10, dt: float = 0.1):
        self.name = name
        self.horizon = horizon
        self.dt = dt
        self._step_count = 0

    def step(self, inp: VLAPolicyInput) -> ActionChunk:
        self._step_count += 1

        # Simulate VLA inference latency
        latency = 0.35 + np.random.uniform(0, 0.05)
        time.sleep(latency)

        obs_time = time.time()

        # Generate trajectory
        actions = []
        for i in range(self.horizon):
            t = obs_time + i * self.dt
            action = np.array([
                0.1 * np.sin(2 * np.pi * 0.5 * t + j * 0.5)
                for j in range(7)
            ])
            actions.append(action)

        if self._step_count % 5 == 0:
            triggered = "replan" if inp.replan_request else "rate"
            print(f"[{self.name}] Chunk #{self._step_count} ({triggered})")

        return ActionChunk(
            action_chunk=actions,
            timestamp=obs_time,
            dt=self.dt,
            status="running"
        )

@io
class ActionBufferOutput:
    """Output from ActionBuffer - single action + optional replan request."""
    action: Optional[np.ndarray] = None
    chunk_index: int = 0
    status: str = ""
    replan_request: bool = False  # True if chunk exhausted


class ActionBufferFlow(Flow[ActionChunk, ActionBufferOutput]):
    """
    Temporal alignment buffer between VLA (2.5Hz) and Controller (200Hz).

    Input: ActionChunk (directly from VLAChunkingPolicyFlow output)
    Output: ActionBufferOutput (single action + replan_request)

    Features:
    - Fast-forward: k = (t_now - t_obs) / dt
    - Linear interpolation for smooth motion
    - Requests VLA replan when chunk expires (via replan_request output)
    """
    def __init__(self, name: str = "ActionBufferFlow", dt: float = 0.1):
        self.name = name
        self.dt = dt
        self._last_chunk: Optional[ActionChunk] = None
        self._step_count = 0
        self._last_log_time = 0.0
        self._requested_replan = False

    def init(self):
        self._last_log_time = time.time()

    def step(self, inp: ActionChunk) -> ActionBufferOutput:
        # Store new chunk
        if inp.action_chunk:
            self._last_chunk = inp
            self._requested_replan = False

        # No chunk yet
        if self._last_chunk is None or not self._last_chunk.action_chunk:
            return ActionBufferOutput(action=None, chunk_index=-1, replan_request=False)

        # Temporal alignment
        now = time.time()
        chunk_start = self._last_chunk.timestamp
        dt = self._last_chunk.dt if self._last_chunk.dt > 0 else self.dt
        delta_t = now - chunk_start

        if delta_t < 0:
            return ActionBufferOutput(action=None, chunk_index=-1, replan_request=False)

        k_float = delta_t / dt
        k = int(k_float)
        chunk = self._last_chunk.action_chunk

        # Check if chunk exhausted → request VLA replan
        if k >= len(chunk):
            if not self._requested_replan:
                self._requested_replan = True
                return ActionBufferOutput(
                    action=np.array(chunk[-1]) if chunk else None,
                    chunk_index=len(chunk) - 1,
                    status="expired",
                    replan_request=True
                )
            return ActionBufferOutput(action=None, chunk_index=k, replan_request=False)

        # Interpolation
        if k + 1 < len(chunk):
            alpha = k_float - k
            action = (1 - alpha) * np.array(chunk[k]) + alpha * np.array(chunk[k + 1])
        else:
            action = np.array(chunk[k])

        self._step_count += 1

        if self._step_count % 100 == 0:
            elapsed = now - self._last_log_time
            rate = 100 / elapsed if elapsed > 0 else 0
            print(f"[{self.name}] Steps: {self._step_count}, Rate: {rate:.1f} Hz, k={k}")
            self._last_log_time = now

        return ActionBufferOutput(
            action=action,
            chunk_index=k,
            status="active",
            replan_request=False
        )


# --- Controller Input Type ---

@io
class ControllerInput:
    """Input to controller - single action from buffer."""
    action: Optional[np.ndarray] = None
    chunk_index: int = 0
    status: str = ""


class RobotControllerSink(Flow[ControllerInput, ControlCommand]):
    """Robot controller at 200Hz. Receives single actions from ActionBuffer."""
    def __init__(self, name: str = "RobotControllerSink"):
        self.name = name
        self._cmd_count = 0
        self._last_log_time = 0.0

    def init(self):
        self._last_log_time = time.time()

    def step(self, inp: ControllerInput) -> ControlCommand:
        if inp.action is None:
            return ControlCommand(action=None)

        self._cmd_count += 1

        if self._cmd_count % 200 == 0:
            now = time.time()
            elapsed = now - self._last_log_time
            rate = 200 / elapsed if elapsed > 0 else 0
            print(f"[{self.name}] Commands: {self._cmd_count}, Rate: {rate:.1f} Hz")
            self._last_log_time = now

        return ControlCommand(action=inp.action)


# ============================================================================
# Pipeline Builder
# ============================================================================

def build_chunking_pipeline() -> Pipeline:
    """
    Build manipulation pipeline with action chunking.

    Reuses existing flows, adds VLAChunkingPolicy and ActionBufferFlow.
    Uses Hybrid clocks for VLA (2.5Hz + replan) and TaskPlanner (1Hz + trigger).
    """
    pipe = Pipeline("manipulation_chunking")

    # --- Source ---
    camera = CameraSourceFlow(name="CameraSourceFlow") @ Rate(30.0)

    # --- Reused Flows ---
    perception = PerceptionFlow(name="PerceptionFlow") @ Trigger("data")
    belief = BeliefUpdaterFlow(name="BeliefUpdaterFlow") @ Trigger("observation")

    # TaskPlanner: Hybrid(1Hz base + Trigger on state changes)
    planner = TaskPlannerFlow(name="TaskPlannerFlow") @ Hybrid(1.0, trigger=["state"])

    # --- VLA Policy: Hybrid(2.5Hz + Trigger on replan_request) ---
    policy = VLAChunkingPolicyFlow(name="VLAChunkingPolicy") @ Hybrid(2.5, trigger=["replan_request"])

    # --- ActionBuffer: 50Hz ---
    action_buffer = ActionBufferFlow(name="ActionBufferFlow", dt=0.1) @ Rate(50.0)

    # --- Controller: 200Hz ---
    controller = RobotControllerSink(name="RobotControllerSink") @ Rate(200.0)

    # --- Monitor: Trigger on status ---
    monitor = ExecutionMonitorFlow(name="ExecutionMonitorFlow") @ Trigger("executor_status")

    # =========================================================================
    # Connections (aligned with demo_manipulation.py)
    # =========================================================================

    # Camera → Perception (data → data)
    pipe.connect(camera, perception, map={"data": "data"}, sync=Latest())

    # Perception → Belief (state → observation, atoms → visible_atoms)
    pipe.connect(perception, belief, map={"state": "observation", "atoms": "visible_atoms"}, sync=Latest())

    # Belief → Planner (belief → state)
    pipe.connect(belief, planner, map={"belief": "state"}, sync=Latest())

    # Planner → Belief (plan → plan)
    pipe.connect(planner, belief, map={"plan": "plan"}, sync=Latest())

    # Planner → Policy (plan → plan)
    pipe.connect(planner, policy, map={"plan": "plan"}, sync=Latest())

    # Belief → Policy (belief → state)
    pipe.connect(belief, policy, map={"belief": "state"}, sync=Latest())

    # Policy → ActionBuffer (ActionChunk: action_chunk, timestamp)
    pipe.connect(policy, action_buffer,
                 map={"action_chunk": "action_chunk", "timestamp": "timestamp"},
                 sync=Latest())

    # ActionBuffer → Controller (action, chunk_index, status)
    pipe.connect(action_buffer, controller,
                 map={"action": "action"},
                 sync=Latest())

    # ActionBuffer → Policy (replan_request - feedback loop!)
    pipe.connect(action_buffer, policy, map={"replan_request": "replan_request"}, sync=Latest())

    # Note: Policy→Belief action feedback removed for VLA architecture
    # (VLA outputs ActionChunk, not Action - see ActionBuffer for action streaming)

    # Belief → Monitor (belief → state)
    pipe.connect(belief, monitor, map={"belief": "state"}, sync=Latest())

    # Policy → Monitor (status → executor_status)
    pipe.connect(policy, monitor, map={"status": "executor_status"}, sync=Latest())

    # Monitor → Planner (replan_config → replan_config)
    pipe.connect(monitor, planner, map={"replan_config": "replan_config"}, sync=Latest())

    return pipe


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Manipulation Pipeline with Action Chunking")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    args = parser.parse_args()

    retriever.init(
        backend="dora",
        backend_config={
            "dora_timeout": 10,
            "rerun_config": {"connect_addr": "127.0.0.1:9876"},
        }
    )

    pipe = build_chunking_pipeline()

    pipe.visualize("viz-manipulation-chunking.html")

    print("=" * 70)
    print("Manipulation Pipeline with Action Chunking")
    print("=" * 70)
    print()
    print("  Hybrid Clocks:")
    print("    • TaskPlanner:  1Hz + Trigger(state)")
    print("    • VLAPolicy:    2.5Hz + Trigger(replan_request)")
    print()
    print("  ActionBuffer → VLAPolicy feedback loop for chunk expiry!")
    print()
    print("=" * 70)

    pipe.run(duration=args.duration)


if __name__ == "__main__":
    main()
