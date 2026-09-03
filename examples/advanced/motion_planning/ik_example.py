"""
Inverse kinematics (IK) example using RoboPlan.

The flow here is:
    - Cartesian pose target generator
    - IK solver
    - Display flow (``ViserSink``)

The payloads and the display flow live in ``flows.py``.
"""

import time
import numpy as np
import xacro

from retriever.flow import Flow, Rate, Pipeline
from retriever.flow.clock import Trigger
from roboplan.core import Scene, CartesianConfiguration, JointConfiguration
from roboplan.example_models import get_package_models_dir, get_package_share_dir
from roboplan.simple_ik import SimpleIk, SimpleIkOptions

from examples.advanced.motion_planning.flows import (
    ARM_GROUP,
    BASE_FRAME,
    HOME_POSITIONS,
    MODEL_DIR,
    TIP_FRAME,
    CartesianTarget,
    JointTarget,
    ViserSink,
)


## Flow classes for pipeline components

class PoseGenerator(Flow[None, CartesianTarget]):
    def reset(self):
        self._start_tform = np.array([
            [1.0, 0.0, 0.0, 0.307],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.59],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._counter = 0
        self._step = 0.05

    def run(self, _):
        offset = np.zeros((4,4))
        offset[1][3] = 0.15 * np.sin(self._counter * self._step)
        offset[2][3] = 0.15 * np.cos(self._counter * self._step)
        target_tform = self._start_tform + offset
        self._counter += 1

        return CartesianTarget(
            base_frame=BASE_FRAME,
            tip_frame=TIP_FRAME,
            tform=target_tform,
        )

class IkSolver(Flow[CartesianTarget, JointTarget]):
    def reset(self):
        # Load model into RoboPlan scene and make an IK solver
        models_dir = get_package_models_dir() / MODEL_DIR
        self._scene = Scene(
            "retriever_scene",
            urdf=xacro.process_file(models_dir / "fr3.urdf").toxml(),
            srdf=xacro.process_file(models_dir / "fr3.srdf").toxml(),
            package_paths=[get_package_share_dir()],
            yaml_config_path=models_dir / "fr3_config.yaml",
        )
        self._scene.setJointPositions(np.array(HOME_POSITIONS))
        self._joint_names = self._scene.getJointNames()
        self._q_indices = self._scene.getJointGroupInfo(ARM_GROUP).q_indices

        self._ik_options = SimpleIkOptions()
        self._ik_options.group_name = ARM_GROUP
        self._ik_options.max_iters = 50
        self._ik_options.step_size = 0.2
        self._ik_options.check_collisions = True
        self._ik_options.max_restarts = 5
        self._ik_solver = SimpleIk(self._scene, self._ik_options)


    def run(self, input: CartesianTarget):
        if not input.base_frame or not input.tip_frame or input.tform is None:
            return

        start = JointConfiguration()
        q_full = self._scene.getCurrentJointPositions()
        start.positions = q_full[self._q_indices]

        goal = CartesianConfiguration()
        goal.base_frame = input.base_frame
        goal.tip_frame = input.tip_frame
        goal.tform = input.tform

        solution = JointConfiguration()

        t_start = time.time()
        result = self._ik_solver.solveIk(goal, start, solution)
        dt = time.time() - t_start
        if result:
            print(f"IK solved in {dt * 1.0e6} us, solution: {solution.positions}")
            q_full[self._q_indices] = solution.positions
            self._scene.setJointPositions(q_full)
            return JointTarget(
                joint_names=self._joint_names,
                joint_positions=q_full
            )
        else:
            print("Failed to solve IK.")
            return JointTarget()


## Assembling the pipeline

def main():
    pipe = Pipeline("motion_planning_demo")
    with pipe:
        pose_gen = PoseGenerator() @ Rate(hz=20)
        inv_kin = IkSolver() @ Trigger("base_frame", "tip_frame", "tform")
        viser_viz = ViserSink() @ Trigger("joint_names", "joint_positions")

        pose_gen >> inv_kin >> viser_viz

    pipe.run(backend="dora", duration=30.0, blocking=True)


if __name__ == "__main__":
    main()
