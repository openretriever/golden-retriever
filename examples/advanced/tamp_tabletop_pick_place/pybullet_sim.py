from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from domain import GroundAction
from motion_refiner import MotionSegment
from scene import Pose2D, TabletopScene


SimMode = Literal["pybullet-direct", "pybullet-gui"]


@dataclass(frozen=True)
class SimConfig:
    mode: SimMode = "pybullet-direct"
    path_steps: int = 32
    gui_sleep_s: float = 1.0 / 240.0


class PyBulletTabletopSimulator:
    def __init__(self, scene: TabletopScene, config: SimConfig) -> None:
        try:
            import pybullet as p
        except ImportError as exc:
            raise RuntimeError(
                "PyBullet is required for simulator-backed TAMP runs. "
                "Install `pybullet` in the Python environment used to launch the demo."
            ) from exc

        self._p = p
        self._config = config
        connection_mode = p.GUI if config.mode == "pybullet-gui" else p.DIRECT
        self._client_id = p.connect(connection_mode)
        if self._client_id < 0:
            raise RuntimeError("Failed to connect to PyBullet.")

        if config.mode == "pybullet-gui":
            self._configure_gui()

        self._table_top_z = 0.0
        self._block_half_extents = (0.02, 0.02, 0.02)
        self._tool_radius = 0.015
        self._transit_z = 0.16
        self._contact_z = self._table_top_z + self._block_half_extents[2] + self._tool_radius

        self._block_id: int | None = None
        self._tool_id: int | None = None
        self._held_object: str | None = None

        self._reset_world(scene)

    def _configure_gui(self) -> None:
        p = self._p
        # Hide the default Bullet debug panes so the scene is actually visible.
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        time.sleep(0.15)

    def close(self) -> None:
        if self._client_id >= 0:
            self._p.disconnect(self._client_id)
            self._client_id = -1

    def execute(self, action: GroundAction, segment: MotionSegment, scene: TabletopScene) -> None:
        if action.name == "Pick":
            self._animate_pick(action.args[0], segment, scene)
        elif action.name == "Place":
            self._animate_place(action.args[0], segment)
        else:
            raise ValueError(f"Unsupported action type: {action.name}")

    def _reset_world(self, scene: TabletopScene) -> None:
        p = self._p
        if self._config.mode == "pybullet-gui":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        self._create_table()
        self._create_regions(scene)
        self._create_block(scene)
        self._create_obstacle(scene)
        self._create_tool(scene)

        if self._config.mode == "pybullet-gui":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=45.0,
                cameraPitch=-60.0,
                cameraTargetPosition=[0.32, 0.0, 0.02],
            )
            for _ in range(4):
                p.stepSimulation()
                time.sleep(self._config.gui_sleep_s)

    def _create_table(self) -> None:
        p = self._p
        half_extents = [0.40, 0.28, 0.02]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.55, 0.42, 0.28, 1.0],
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0.32, 0.0, -half_extents[2]],
        )

    def _create_regions(self, scene: TabletopScene) -> None:
        self._create_region_box(scene.start_region.center, [0.25, 0.75, 0.35, 0.45])
        self._create_region_box(scene.goal_region.center, [0.35, 0.45, 0.95, 0.45])

    def _create_region_box(self, pose: Pose2D, color: list[float]) -> None:
        p = self._p
        half_extents = [0.055, 0.055, 0.001]
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual,
            basePosition=[pose.x, pose.y, self._table_top_z + half_extents[2]],
        )

    def _create_block(self, scene: TabletopScene) -> None:
        p = self._p
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=list(self._block_half_extents),
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=list(self._block_half_extents),
            rgbaColor=[0.85, 0.2, 0.2, 1.0],
        )
        self._block_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=self._block_xyz(scene.block_pose),
        )

    def _create_obstacle(self, scene: TabletopScene) -> None:
        if scene.obstacle is None:
            return
        p = self._p
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=scene.obstacle.radius,
            height=0.14,
        )
        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=scene.obstacle.radius,
            length=0.14,
            rgbaColor=[0.88, 0.88, 0.92, 1.0],
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[scene.obstacle.center.x, scene.obstacle.center.y, 0.07],
        )

    def _create_tool(self, scene: TabletopScene) -> None:
        p = self._p
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=self._tool_radius)
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self._tool_radius,
            rgbaColor=[0.1, 0.45, 0.95, 1.0],
        )
        self._tool_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=self._tool_xyz(scene.block_pose, self._transit_z),
        )

    def _animate_pick(self, object_name: str, segment: MotionSegment, scene: TabletopScene) -> None:
        del scene
        self._move_tool(segment.approach_pose, self._transit_z, carrying=False)
        self._move_tool(segment.target_pose, self._contact_z, carrying=False)
        self._held_object = object_name
        self._move_tool(segment.retreat_pose, self._transit_z, carrying=True)

    def _animate_place(self, object_name: str, segment: MotionSegment) -> None:
        if self._held_object != object_name:
            raise ValueError(
                f"Simulator expected held object {self._held_object!r}, got {object_name!r}"
            )
        self._move_tool(segment.approach_pose, self._transit_z, carrying=True)
        self._move_tool(segment.target_pose, self._contact_z, carrying=True)
        self._set_block_pose(segment.target_pose, z=self._block_half_extents[2])
        self._held_object = None
        self._move_tool(segment.retreat_pose, self._transit_z, carrying=False)

    def _move_tool(self, target_pose: Pose2D, z: float, *, carrying: bool) -> None:
        assert self._tool_id is not None
        p = self._p
        start_xyz = p.getBasePositionAndOrientation(self._tool_id)[0]
        end_xyz = self._tool_xyz(target_pose, z)
        for step in range(1, self._config.path_steps + 1):
            alpha = step / self._config.path_steps
            xyz = [
                start_xyz[i] + alpha * (end_xyz[i] - start_xyz[i])
                for i in range(3)
            ]
            p.resetBasePositionAndOrientation(self._tool_id, xyz, [0.0, 0.0, 0.0, 1.0])
            if carrying and self._block_id is not None:
                block_xyz = [xyz[0], xyz[1], max(self._block_half_extents[2], xyz[2] - self._tool_radius)]
                p.resetBasePositionAndOrientation(
                    self._block_id,
                    block_xyz,
                    [0.0, 0.0, 0.0, 1.0],
                )
            p.stepSimulation()
            if self._config.mode == "pybullet-gui":
                time.sleep(self._config.gui_sleep_s)

    def _set_block_pose(self, pose: Pose2D, z: float) -> None:
        if self._block_id is None:
            return
        self._p.resetBasePositionAndOrientation(
            self._block_id,
            [pose.x, pose.y, z],
            [0.0, 0.0, 0.0, 1.0],
        )

    def _block_xyz(self, pose: Pose2D) -> list[float]:
        return [pose.x, pose.y, self._block_half_extents[2]]

    def _tool_xyz(self, pose: Pose2D, z: float) -> list[float]:
        return [pose.x, pose.y, z]
