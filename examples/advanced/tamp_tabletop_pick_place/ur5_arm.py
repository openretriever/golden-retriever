from __future__ import annotations

import time
from dataclasses import dataclass
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
    path_steps: int = 48
    gui_sleep_s: float = 1.0 / 60.0
    control_substeps: int = 1


class UR5SuctionArm:
    def __init__(self, p: object, config: UR5SuctionArmConfig) -> None:
        self._p = p
        self._config = config
        self._joint_indices = [2, 3, 4, 5, 6, 7]
        self._tool0_link_index = 9
        self._tool_tip_link_index = 10
        self._downward_orientation = p.getQuaternionFromEuler((0.0, 0.0, 0.0))

        self._robot_id = self._load_urdf_strict(UR5_URDF_PATH, useFixedBase=True)
        self._suction_base_id = self._load_visual_urdf(SUCTION_BASE_URDF_PATH)
        self._suction_head_id = self._load_visual_urdf(SUCTION_HEAD_URDF_PATH)
        self.reset_ready_pose()
        self._sync_tool_visuals()

    @property
    def robot_id(self) -> int:
        return self._robot_id

    def reset_ready_pose(self) -> None:
        for joint_index, value in zip(self._joint_indices, UR5_HOME_JOINTS):
            self._p.resetJointState(self._robot_id, joint_index, float(value))
        for joint_index, value in zip(self._joint_indices, UR5_READY_JOINTS):
            self._p.resetJointState(self._robot_id, joint_index, float(value))
        self._sync_tool_visuals()

    def move_tip_linear(
        self,
        target_xyz: tuple[float, float, float],
        *,
        on_step: Callable[[tuple[float, float, float]], None] | None = None,
    ) -> None:
        start_xyz = np.array(self.current_suction_tip_xyz(), dtype=float)
        target_xyz_arr = np.array(target_xyz, dtype=float)

        for step in range(1, self._config.path_steps + 1):
            alpha = step / self._config.path_steps
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            waypoint_xyz = start_xyz + smooth_alpha * (target_xyz_arr - start_xyz)
            joint_targets = self._solve_tip_ik(tuple(waypoint_xyz.tolist()))
            for joint_index, value in zip(self._joint_indices, joint_targets):
                self._p.resetJointState(self._robot_id, joint_index, float(value))

            for _ in range(self._config.control_substeps):
                self._p.stepSimulation()
                self._sync_tool_visuals()
                if on_step is not None:
                    on_step(tuple(self.current_suction_tip_xyz()))
                if self._config.gui_sleep_s > 0.0:
                    time.sleep(self._config.gui_sleep_s)

    def current_joint_positions(self) -> np.ndarray:
        return np.array(
            [self._p.getJointState(self._robot_id, joint_index)[0] for joint_index in self._joint_indices],
            dtype=float,
        )

    def current_tcp_xyz(self) -> list[float]:
        return self.current_suction_tip_xyz()

    def current_tcp_orientation(self) -> tuple[float, float, float, float]:
        return self._link_orientation(self._tool_tip_link_index)

    def current_suction_tip_xyz(self) -> list[float]:
        return self._link_position(self._tool_tip_link_index)

    def _solve_tip_ik(self, xyz: tuple[float, float, float]) -> np.ndarray:
        joint_targets = self._p.calculateInverseKinematics(
            bodyUniqueId=self._robot_id,
            endEffectorLinkIndex=self._tool_tip_link_index,
            targetPosition=xyz,
            targetOrientation=self._downward_orientation,
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],
            restPoses=np.float32(UR5_HOME_JOINTS).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        return np.array(joint_targets[:6], dtype=float)

    def _load_visual_urdf(self, path: Path) -> int:
        body_id = self._load_urdf_strict(path, useFixedBase=True)
        for link_index in range(-1, self._p.getNumJoints(body_id)):
            self._p.setCollisionFilterGroupMask(body_id, link_index, 0, 0)
        return body_id

    def _sync_tool_visuals(self) -> None:
        tool0_xyz = np.array(self._link_position(self._tool0_link_index), dtype=float)
        tip_xyz = np.array(self._link_position(self._tool_tip_link_index), dtype=float)
        direction = tip_xyz - tool0_xyz
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            direction = np.array([0.0, 0.0, -1.0], dtype=float)
        else:
            direction = direction / norm

        orientation = self._quat_from_z_axis(direction)
        head_base_xyz = (tip_xyz - direction * 0.029).tolist()

        self._p.resetBasePositionAndOrientation(
            self._suction_base_id,
            tool0_xyz.tolist(),
            orientation,
        )
        self._p.resetBasePositionAndOrientation(
            self._suction_head_id,
            head_base_xyz,
            orientation,
        )

    def _link_position(self, link_index: int) -> list[float]:
        link_state = self._p.getLinkState(self._robot_id, link_index)
        return list(link_state[4])

    def _link_orientation(self, link_index: int) -> tuple[float, float, float, float]:
        link_state = self._p.getLinkState(self._robot_id, link_index)
        return tuple(link_state[5])

    def _quat_from_z_axis(self, direction: np.ndarray) -> tuple[float, float, float, float]:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        direction = direction / np.linalg.norm(direction)
        dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))

        if dot >= 1.0 - 1e-8:
            return (0.0, 0.0, 0.0, 1.0)
        if dot <= -1.0 + 1e-8:
            return tuple(self._p.getQuaternionFromEuler((np.pi, 0.0, 0.0)))

        axis = np.cross(z_axis, direction)
        axis = axis / np.linalg.norm(axis)
        angle = float(np.arccos(dot))
        half_angle = angle / 2.0
        sin_half = float(np.sin(half_angle))
        return (
            float(axis[0] * sin_half),
            float(axis[1] * sin_half),
            float(axis[2] * sin_half),
            float(np.cos(half_angle)),
        )

    def _load_urdf_strict(self, path: Path, *args, **kwargs) -> int:
        body_id = self._p.loadURDF(str(path), *args, **kwargs)
        if body_id < 0:
            raise RuntimeError(f"Failed to load URDF: {path}")
        return body_id
