from __future__ import annotations

import time
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Callable

import numpy as np


ASSETS_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "golden_retriever"
    / "envs"
    / "ravens"
    / "envs"
    / "assets"
)
UR5_URDF_PATH = ASSETS_ROOT / "ur5" / "ur5.urdf"
SUCTION_BASE_URDF_PATH = ASSETS_ROOT / "ur5" / "suction" / "suction-base.urdf"
SUCTION_HEAD_URDF_PATH = ASSETS_ROOT / "ur5" / "suction" / "suction-head.urdf"

UR5_HOME_JOINTS = np.array([-1.0, -0.5, 0.5, -0.5, -0.5, 0.0]) * np.pi
UR5_READY_JOINTS = np.array(
    [-3.5653608, -1.1199822, 0.02722096, -0.50724125, -1.5674078, -0.4235661]
)


@dataclass(frozen=True)
class UR5SuctionArmConfig:
    path_steps: int = 32
    gui_sleep_s: float = 1.0 / 60.0


class UR5SuctionArm:
    def __init__(self, p: object, config: UR5SuctionArmConfig) -> None:
        self._p = p
        self._config = config
        self._joint_indices = [2, 3, 4, 5, 6, 7]
        self._tool_link_index = 10
        self._tool_parent_link = 9
        self._downward_orientation = p.getQuaternionFromEuler((0.0, 0.0, 0.0))

        self._robot_id = self._load_urdf_strict(UR5_URDF_PATH, useFixedBase=True)
        self._suction_base_id = self._load_urdf_strict(
            SUCTION_BASE_URDF_PATH,
            [0.487, 0.109, 0.438],
            p.getQuaternionFromEuler((pi, 0.0, 0.0)),
        )
        self._suction_head_id = self._load_urdf_strict(
            SUCTION_HEAD_URDF_PATH,
            [0.487, 0.109, 0.347],
            p.getQuaternionFromEuler((pi, 0.0, 0.0)),
        )
        self._attach_suction_geometry()
        self.reset_ready_pose()
        tcp_xyz = self.current_tcp_xyz()
        tip_xyz = self.current_suction_tip_xyz()
        self._tcp_to_tip_offset = tuple(tip_xyz[i] - tcp_xyz[i] for i in range(3))

    @property
    def robot_id(self) -> int:
        return self._robot_id

    def reset_ready_pose(self) -> None:
        for joint_index, value in zip(self._joint_indices, UR5_HOME_JOINTS):
            self._p.resetJointState(self._robot_id, joint_index, float(value))
        for joint_index, value in zip(self._joint_indices, UR5_READY_JOINTS):
            self._p.resetJointState(self._robot_id, joint_index, float(value))

    def move_tip_linear(
        self,
        target_xyz: tuple[float, float, float],
        *,
        on_step: Callable[[tuple[float, float, float]], None] | None = None,
    ) -> None:
        start_xyz = self.current_suction_tip_xyz()
        end_xyz = list(target_xyz)
        for step in range(1, self._config.path_steps + 1):
            alpha = step / self._config.path_steps
            xyz = [
                start_xyz[i] + alpha * (end_xyz[i] - start_xyz[i])
                for i in range(3)
            ]
            self._set_tip_pose(tuple(xyz))
            if on_step is not None:
                on_step(tuple(xyz))
            self._p.stepSimulation()
            if self._config.gui_sleep_s > 0.0:
                time.sleep(self._config.gui_sleep_s)

    def current_tcp_xyz(self) -> list[float]:
        link_state = self._p.getLinkState(self._robot_id, self._tool_link_index)
        return list(link_state[4])

    def current_suction_tip_xyz(self) -> list[float]:
        link_state = self._p.getLinkState(self._suction_head_id, 0)
        return list(link_state[4])

    def _set_tip_pose(self, xyz: tuple[float, float, float]) -> None:
        tcp_xyz = tuple(xyz[i] - self._tcp_to_tip_offset[i] for i in range(3))
        self._set_tcp_pose(tcp_xyz)

    def _set_tcp_pose(self, xyz: tuple[float, float, float]) -> None:
        joint_targets = self._p.calculateInverseKinematics(
            bodyUniqueId=self._robot_id,
            endEffectorLinkIndex=self._tool_link_index,
            targetPosition=xyz,
            targetOrientation=self._downward_orientation,
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],
            restPoses=np.float32(UR5_HOME_JOINTS).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        for joint_index, value in zip(self._joint_indices, joint_targets[:6]):
            self._p.resetJointState(self._robot_id, joint_index, float(value))

    def _attach_suction_geometry(self) -> None:
        base_constraint = self._p.createConstraint(
            parentBodyUniqueId=self._robot_id,
            parentLinkIndex=self._tool_parent_link,
            childBodyUniqueId=self._suction_base_id,
            childLinkIndex=-1,
            jointType=self._p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01),
        )
        head_constraint = self._p.createConstraint(
            parentBodyUniqueId=self._robot_id,
            parentLinkIndex=self._tool_parent_link,
            childBodyUniqueId=self._suction_head_id,
            childLinkIndex=-1,
            jointType=self._p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08),
        )
        self._p.changeConstraint(base_constraint, maxForce=50)
        self._p.changeConstraint(head_constraint, maxForce=50)

    def _load_urdf_strict(self, path: Path, *args, **kwargs) -> int:
        body_id = self._p.loadURDF(str(path), *args, **kwargs)
        if body_id < 0:
            raise RuntimeError(f"Failed to load URDF: {path}")
        return body_id
