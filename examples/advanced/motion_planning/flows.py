"""Shared payloads and display flow for the RoboPlan motion planning examples.

Both ``ik_example.py`` and ``motion_tracking_example.py`` drive the Franka FR3
model bundled with RoboPlan's example models and display it with Viser. The
payloads they exchange and the ``ViserSink`` display flow live here; the IK and
motion planning flows stay with their examples.

The ``demo-ik`` and ``demo-motion-track`` Pixi tasks set ``PYTHONPATH=.`` so this
module imports as ``examples.advanced.motion_planning.flows``.
"""

import time
from dataclasses import dataclass

import numpy as np
import pinocchio as pin
import xacro
from pinocchio.visualize import ViserVisualizer
from retriever.flow import Flow, io
from roboplan.example_models import get_package_models_dir, get_package_share_dir

# Franka FR3 model bundled with RoboPlan's example models.
MODEL_DIR = "franka_robot_model"
ARM_GROUP = "fr3_arm"
BASE_FRAME = "fr3_link0"
TIP_FRAME = "fr3_hand"
# Full configuration (7 arm joints + gripper) used as the starting pose everywhere.
HOME_POSITIONS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.01)


## Payloads


@io
@dataclass
class CartesianTarget:
    """Target pose of ``tip_frame`` relative to ``base_frame``, as a 4x4 transform."""

    base_frame: str
    tip_frame: str
    tform: np.ndarray | None = None


@io
@dataclass
class JointTarget:
    """Instantaneous joint state of the full model, ordered like ``joint_names``."""

    joint_names: list[str]
    joint_positions: np.ndarray


@io
@dataclass
class VisualizationInput:
    """Everything ``ViserSink`` can display.

    ``joint_names`` and ``joint_positions`` carry the instantaneous robot state.
    ``traj_joint_positions`` and ``traj_times`` are optional and carry the latest
    full trajectory (one row of positions per time) so its end effector path can
    be drawn.
    """

    joint_names: list[str]
    joint_positions: np.ndarray
    traj_joint_positions: np.ndarray
    traj_times: np.ndarray


## Flows


class ViserSink(Flow[VisualizationInput, None]):
    """Display the robot in a Viser viewer.

    Every input updates the robot pose from ``joint_positions``. When the input
    also carries a trajectory, the end effector path of that trajectory is drawn
    as a polyline. Trajectories are told apart by their start time, so the same
    trajectory arriving repeatedly is drawn once.
    """

    def __init__(self, *, path_frame: str = TIP_FRAME, path_name: str = "trajectory/path") -> None:
        super().__init__()
        self.path_frame = path_frame
        self.path_name = path_name
        self._viz = None

    def init_config(self) -> dict:
        return {"path_frame": self.path_frame, "path_name": self.path_name}

    def reset(self) -> None:
        # Create Pinocchio model for visualization
        models_dir = get_package_models_dir() / MODEL_DIR
        package_dirs = [get_package_share_dir()]
        urdf_xml = xacro.process_file(models_dir / "fr3.urdf").toxml()

        self._model = pin.buildModelFromXML(urdf_xml, mimic=True)
        collision_model = pin.buildGeomFromUrdfString(
            self._model, urdf_xml, pin.GeometryType.COLLISION, package_dirs=package_dirs
        )
        visual_model = pin.buildGeomFromUrdfString(
            self._model, urdf_xml, pin.GeometryType.VISUAL, package_dirs=package_dirs
        )
        self._data = self._model.createData()
        self._path_frame_id = self._model.getFrameId(self.path_frame)
        self._last_traj_start_time = None

        if self._viz is None:
            # Open the viewer once; later resets only move the robot back home.
            self._viz = ViserVisualizer(self._model, collision_model, visual_model)
            self._viz.initViewer(open=True, loadModel=True)
        self._viz.display(np.array(HOME_POSITIONS))
        time.sleep(0.1)  # Give the viewer a moment to render

    def run(self, input: VisualizationInput) -> None:
        if input.joint_positions is None:
            return
        if input.traj_joint_positions is not None and input.traj_times is not None:
            self._draw_path(input.traj_joint_positions, input.traj_times)
        self._viz.display(input.joint_positions)

    def _draw_path(self, traj_joint_positions: np.ndarray, traj_times: np.ndarray) -> None:
        start_time = traj_times[0]
        if start_time == self._last_traj_start_time:
            return
        self._last_traj_start_time = start_time

        # Trajectories only cover the arm joints, which come first in the model.
        q = np.array(HOME_POSITIONS)
        translations = []
        for positions in traj_joint_positions:
            q[: len(positions)] = positions
            pin.framesForwardKinematics(self._model, self._data, q)
            translations.append(self._data.oMf[self._path_frame_id].translation.copy())

        self._viz.viewer.scene.add_line_segments(
            self.path_name,
            points=np.array([list(pair) for pair in zip(translations, translations[1:])]),
            colors=(0, 150, 100),
            line_width=3.0,
        )
