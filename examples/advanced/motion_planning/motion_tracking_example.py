"""
Motion tracking example using RoboPlan.

The flow here is:
    - RRT based motion planner generating trajectories at random times
    - Trackers that follow the trajectories received
    - Display flow (``ViserSink``)

The joint-space payloads and the display flow live in ``flows.py``.
"""

from dataclasses import dataclass
import time
import numpy as np
import xacro

from retriever.flow import Flow, io, Pipeline, Hybrid, Latest
from retriever.flow.clock import Trigger
from roboplan.core import Scene, JointConfiguration
from roboplan.example_models import get_package_models_dir, get_package_share_dir
from roboplan.rrt import RRTOptions, RRT
from roboplan.toppra import PathParameterizerTOPPRA, SplineFittingMode, TOPPRAOptions

from examples.advanced.motion_planning.flows import (
    ARM_GROUP,
    HOME_POSITIONS,
    MODEL_DIR,
    JointTarget,
    VisualizationInput,
    ViserSink,
)

## Dataclasses for communication

@io
@dataclass
class JointTrajectory:
    joint_names: list[str]
    joint_positions: np.ndarray
    times: np.ndarray


## Flow classes for pipeline components

class MotionPlanner(Flow[JointTarget, JointTrajectory]):
    def reset(self):
        # Load model into RoboPlan scene and make a motion planner
        models_dir = get_package_models_dir() / MODEL_DIR
        self._scene = Scene(
            "retriever_scene",
            urdf=xacro.process_file(models_dir / "fr3.urdf").toxml(),
            srdf=xacro.process_file(models_dir / "fr3.srdf").toxml(),
            package_paths=[get_package_share_dir()],
            yaml_config_path=models_dir / "fr3_config.yaml",
        )
        self._scene.setJointPositions(np.array(HOME_POSITIONS))
        self._q_indices = self._scene.getJointGroupInfo(ARM_GROUP).q_indices

        # Set up an RRT and perform path planning.
        options = RRTOptions()
        options.max_connection_distance = 5.0
        options.group_name = ARM_GROUP
        self._rrt = RRT(self._scene, options)

        # Set up a trajectory timing algorithm (TOPP-RA).
        self._toppra = PathParameterizerTOPPRA(self._scene, ARM_GROUP)

        self._last_planning_time = time.time() - 100.0  # Force planning on the first step

    def run(self, input: JointTarget):
        # Update state
        if input.joint_positions is not None:
            self._scene.setJointPositions(input.joint_positions)

        # Find a random time to replan.
        now = time.time()
        if (now - self._last_planning_time < np.random.uniform(2.0, 5.0)):
            return

        print("Replanning...")
        q_full = self._scene.getCurrentJointPositions()

        # Path planning
        start = JointConfiguration()
        start.positions = q_full[self._q_indices]

        goal = JointConfiguration()
        goal.positions = self._scene.randomCollisionFreePositions()[self._q_indices]

        path = self._rrt.plan(start, goal)
        if path is None:
            print("Planning failed :(")
            return
        self._last_planning_time = now

        # Trajectory timing
        traj_dt = 0.01  # Match the rate configured for the flow, is there an easier way to get this info?
        traj = self._toppra.generate(path, TOPPRAOptions(dt=traj_dt, mode=SplineFittingMode.Adaptive))
        return JointTrajectory(
            joint_names=traj.joint_names,
            joint_positions=traj.positions,
            times=now + np.asarray(traj.times),
        )


class TrajTracker(Flow[JointTrajectory, JointTarget]):
    def reset(self):
        self._last_traj_start_time = None
        self._waypoint_idx = 0
        self._q = np.array(HOME_POSITIONS)

    def run(self, input: JointTrajectory):
        if input.joint_positions is None or input.times is None:
            return

        joint_positions = input.joint_positions.copy()
        start_time = input.times[0]

        if self._last_traj_start_time != start_time:
            print("Received new trajectory!")
            self._waypoint_idx = 0
            self._last_traj_start_time = start_time
        elif self._waypoint_idx >= len(joint_positions) - 1:
            print("Finished executing trajectory!")
            self._waypoint_idx = -1
            return
        elif self._waypoint_idx >= 0:
            self._waypoint_idx += 1
        else:
            # Waypoint idx is -1 meaning inactive
            return

        self._q[:7] = joint_positions[self._waypoint_idx]  # Include gripper states
        output = JointTarget(
            joint_names=input.joint_names,
            joint_positions=self._q,
        )
        return output


class VizAggregator(Flow[VisualizationInput, VisualizationInput]):
    """Re-emit the latest robot state and trajectory together for the display flow."""

    def run(self, input: VisualizationInput):
        output =  VisualizationInput()
        if input._has_signal("joint_names"):
            output.joint_names = input._get_signal("joint_names")
        if input._has_signal("joint_positions"):
            output.joint_positions = input._get_signal("joint_positions")
        if input._has_signal("traj_joint_positions"):
            output.traj_joint_positions = input._get_signal("traj_joint_positions")
        if input._has_signal("traj_times"):
            output.traj_times = input._get_signal("traj_times")

        return output


## Assembling the pipeline

def main():
    pipe = Pipeline("motion_planning_demo")
    with pipe:
        planner = MotionPlanner() @ Hybrid(hz=10.0, trigger=["joint_positions"])
        traj_tracker = TrajTracker() @ Hybrid(hz=100.0, trigger=["joint_positions"])
        aggregator = VizAggregator() @ Trigger("joint_positions", "traj_joint_positions")
        viser_viz = ViserSink() @ Trigger("joint_positions", "traj_joint_positions", "traj_times")

        pipe.connect(planner, traj_tracker, sync=Latest())
        pipe.connect(
            planner,
            aggregator,
            map={"joint_positions": "traj_joint_positions", "times": "traj_times"},
            sync=Latest(),
        )
        pipe.connect(traj_tracker, aggregator, sync=Latest())
        pipe.connect(aggregator, viser_viz, sync=Latest())
        pipe.connect(traj_tracker, planner, sync=Latest())

    pipe.run(backend="dora", duration=120.0, blocking=True)


if __name__ == "__main__":
    main()
