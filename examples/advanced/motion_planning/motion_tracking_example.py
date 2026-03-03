"""
Motion tracking example using RoboPlan.

The flow here is:
    - RRT based motion planner generating trajectories at random times
    - Trackers that follow the trajectories received
    - Display flow
"""

from dataclasses import dataclass
import time
import numpy as np
import xacro

import pinocchio as pin
from pinocchio.visualize import ViserVisualizer
from retriever.flow import Flow, io, Pipeline, Hybrid, Latest
from retriever.flow.clock import Trigger
from roboplan.core import Scene, JointConfiguration
from roboplan.example_models import get_package_models_dir, get_package_share_dir
from roboplan.rrt import RRTOptions, RRT
from roboplan.toppra import PathParameterizerTOPPRA

## Dataclasses for communication

@io
@dataclass
class JointTrajectory:
    joint_names: list[str]
    joint_positions: np.ndarray
    times: list[float]

@io
@dataclass
class JointTarget:
    joint_names: list[str]
    joint_positions: np.ndarray


@io
@dataclass
class VisualizationInput:
    joint_names: list[str]
    
    # Instantaneous joint target for visualizing robot state
    joint_positions: np.ndarray

    # Full trajectory for visualizing the path
    traj_joint_positions: np.ndarray
    traj_times: list[float]


## Flow classes for pipeline components

class MotionPlanner(Flow[JointTarget, JointTrajectory]):
    def init(self):
        # Load model into RoboPlan scene and make an IK solver
        models_dir = get_package_models_dir()
        self._scene = Scene(
            "retriever_scene",
            urdf=xacro.process_file(models_dir / "franka_robot_model" / "fr3.urdf").toxml(),
            srdf=xacro.process_file(models_dir / "franka_robot_model" / "fr3.srdf").toxml(),
            package_paths=[get_package_share_dir()],
            yaml_config_path=models_dir / "franka_robot_model" / "fr3_config.yaml",
        )

        self._joint_positions = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.01, 0.01])
        self._scene.setJointPositions(self._joint_positions)
        self._joint_group = "fr3_arm"
        self._joint_names = self._scene.getJointNames()
        self._q_indices = self._scene.getJointGroupInfo(self._joint_group).q_indices

        # Set up an RRT and perform path planning.
        options = RRTOptions()
        options.max_connection_distance = 5.0
        options.group_name = self._joint_group
        self._rrt = RRT(self._scene, options)

        # Set up a trajectory timing algorithm (TOPP-RA).
        self._toppra = PathParameterizerTOPPRA(self._scene, self._joint_group)

        self._last_planning_time = time.time() - 100.0  # Force planning on the first step

    def run(self, input: JointTarget):
        # Update state
        if input.joint_positions is not None:
            self._scene.setJointPositions(input.joint_positions)

        # Find a random time to replan.
        now = time.time()
        if (now - self._last_planning_time < np.random.uniform(3.0, 10.0)):
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
        traj = self._toppra.generate(path, traj_dt)
        return JointTrajectory(
            joint_names=traj.joint_names,
            joint_positions=traj.positions,
            times=[now + t for t in traj.times],
        )


class TrajTracker(Flow[JointTrajectory, JointTarget]):
    def init(self):
        self._last_traj_start_time = None
        self._waypoint_idx = 0
        self._q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.01, 0.01])

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
    def __init__(self):
        super().__init__()
        self.target = JointTarget()
        self.trajectory = JointTrajectory()

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


class ViserSink(Flow[VisualizationInput, None]):
    def init(self):
        # Create Pinocchio model for visualization
        models_dir = get_package_models_dir()
        package_paths = [get_package_share_dir()]
        urdf_xml = xacro.process_file(models_dir / "franka_robot_model" / "fr3.urdf").toxml()

        self._model = pin.buildModelFromXML(urdf_xml)
        collision_model = pin.buildGeomFromUrdfString(
            self._model, urdf_xml, pin.GeometryType.COLLISION, package_dirs=package_paths
        )
        visual_model = pin.buildGeomFromUrdfString(
            self._model, urdf_xml, pin.GeometryType.VISUAL, package_dirs=package_paths
        )
        self._data = self._model.createData()
        self._q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.01, 0.01])

        self._viz = ViserVisualizer(self._model, collision_model, visual_model)
        self._viz.initViewer(open=True, loadModel=True)
        self._viz.display(self._q)
        time.sleep(0.1)  # To render

        self._last_traj_start_time = None

    def run(self, input: JointTarget):
        if not input.joint_names:
            return
        
        # Visualize new trajectories as they come in
        if input.traj_joint_positions and input.traj_times:
            if self._last_traj_start_time != input.traj_times[0]:
                self._last_traj_start_time = input.traj_times[0]

                translations = []
                q = self._q
                frame_id = self._model.getFrameId("fr3_hand")
                for positions in input.traj_joint_positions:
                    q[:7] = positions
                    pin.framesForwardKinematics(self._model, self._data, q)
                    tform = self._data.oMf[frame_id]
                    translations.append(tform.translation.copy())

                self._viz.viewer.scene.add_line_segments(
                    "rrt/path",
                    points=np.array([list(pair) for pair in zip(translations, translations[1:])]),
                    colors=(0, 150, 100),
                    line_width=3.0,
                )

        # Visualize the robot state
        self._viz.display(input.joint_positions)


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
