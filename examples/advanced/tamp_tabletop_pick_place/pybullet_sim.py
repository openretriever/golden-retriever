from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from domain import GroundAction
from motion_refiner import MotionSegment
from scene import Pose2D, TabletopScene
from ur5_arm import UR5SuctionArm, UR5SuctionArmConfig

from examples.advanced.shared.pybullet import (
    DebugCameraPose,
    add_debug_label,
    connect_pybullet,
    rendering_disabled,
    set_debug_camera,
    step_gui_frames,
)


SimMode = Literal["pybullet-direct", "pybullet-gui"]


@dataclass(frozen=True)
class SimConfig:
    mode: SimMode = "pybullet-direct"
    path_steps: int = 48
    gui_sleep_s: float = 1.0 / 60.0
    gui_warmup_frames: int = 18


class PyBulletTabletopSimulator:
    def __init__(self, scene: TabletopScene, config: SimConfig) -> None:
        self._config = config
        self._camera = self._camera_for_scene(scene)
        self._p, self._client_id = connect_pybullet(
            gui=config.mode == "pybullet-gui",
            time_step_s=1.0 / 240.0,
        )

        self._table_top_z = 0.0
        self._block_half_extents = (0.02, 0.02, 0.02)
        self._grasp_clearance = 0.002
        self._transit_z = 0.16
        self._contact_z = self._table_top_z + 2 * self._block_half_extents[2] + self._grasp_clearance

        self._block_id: int | None = None
        self._arm: UR5SuctionArm | None = None
        self._held_object: str | None = None

        self._reset_world(scene)

    def close(self) -> None:
        if self._client_id >= 0:
            self._p.disconnect(self._client_id)
            self._client_id = -1

    def hold(self, seconds: float) -> None:
        if self._config.mode != "pybullet-gui" or seconds <= 0.0:
            return

        frames = max(1, round(seconds / max(self._config.gui_sleep_s, 1.0 / 240.0)))
        step_gui_frames(self._p, frames=frames, sleep_s=self._config.gui_sleep_s)

    def execute(self, action: GroundAction, segment: MotionSegment, scene: TabletopScene) -> None:
        if action.name == "Pick":
            self._animate_pick(action.args[0], segment, scene)
        elif action.name == "Place":
            self._animate_place(action.args[0], segment)
        else:
            raise ValueError(f"Unsupported action type: {action.name}")

    def _reset_world(self, scene: TabletopScene) -> None:
        p = self._p
        with rendering_disabled(p, active=self._config.mode == "pybullet-gui"):
            p.resetSimulation()
            p.setGravity(0.0, 0.0, -9.81)
            p.setTimeStep(1.0 / 240.0)

            self._create_table()
            self._create_regions(scene)
            self._create_block(scene)
            self._create_obstacle(scene)
            self._create_arm()
            self._create_labels(scene)

        if self._config.mode == "pybullet-gui":
            set_debug_camera(p, self._camera)
            step_gui_frames(
                p,
                frames=self._config.gui_warmup_frames,
                sleep_s=self._config.gui_sleep_s,
            )
            set_debug_camera(p, self._camera)
            step_gui_frames(p, frames=2, sleep_s=self._config.gui_sleep_s)

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
        half_extents = [0.055, 0.055, 0.002]
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

    def _create_labels(self, scene: TabletopScene) -> None:
        if self._config.mode != "pybullet-gui":
            return

        p = self._p
        add_debug_label(
            p,
            "start",
            (scene.start_region.center.x - 0.03, scene.start_region.center.y, 0.035),
            color=(0.2, 0.9, 0.3),
        )
        add_debug_label(
            p,
            "goal",
            (scene.goal_region.center.x - 0.02, scene.goal_region.center.y, 0.035),
            color=(0.4, 0.5, 1.0),
        )
        add_debug_label(
            p,
            scene.block_name,
            (scene.block_pose.x - 0.03, scene.block_pose.y, 0.06),
            color=(1.0, 0.35, 0.35),
        )
        if scene.obstacle is not None:
            add_debug_label(
                p,
                scene.obstacle.name,
                (scene.obstacle.center.x - 0.02, scene.obstacle.center.y, 0.12),
                color=(0.9, 0.9, 0.9),
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

    def _create_arm(self) -> None:
        self._arm = UR5SuctionArm(
            self._p,
            UR5SuctionArmConfig(
                path_steps=self._config.path_steps,
                gui_sleep_s=self._config.gui_sleep_s if self._config.mode == "pybullet-gui" else 0.0,
            ),
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
        assert self._arm is not None
        on_step = self._move_held_block if carrying and self._block_id is not None else None
        self._arm.move_tip_linear(tuple(self._tool_xyz(target_pose, z)), on_step=on_step)

    def _move_held_block(self, tip_xyz: tuple[float, float, float]) -> None:
        if self._block_id is None:
            return
        block_xyz = [
            tip_xyz[0],
            tip_xyz[1],
            max(
                self._block_half_extents[2],
                tip_xyz[2] - (self._block_half_extents[2] + self._grasp_clearance),
            ),
        ]
        self._p.resetBasePositionAndOrientation(
            self._block_id,
            block_xyz,
            [0.0, 0.0, 0.0, 1.0],
        )

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

    def _camera_for_scene(self, scene: TabletopScene) -> DebugCameraPose:
        points = [
            Pose2D(0.0, 0.0),
            scene.start_region.center,
            scene.goal_region.center,
            scene.block_pose,
        ]
        if scene.obstacle is not None:
            points.append(scene.obstacle.center)

        xs = [point.x for point in points]
        ys = [point.y for point in points]
        x_mid = 0.5 * (min(xs) + max(xs))
        y_mid = 0.5 * (min(ys) + max(ys))
        span = max(max(xs) - min(xs), max(ys) - min(ys))

        return DebugCameraPose(
            distance=max(1.0, 1.8 * span + 0.55),
            yaw=38.0,
            pitch=-50.0,
            target=(x_mid, y_mid, 0.08),
        )
