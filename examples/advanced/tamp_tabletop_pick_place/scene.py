from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float

    def shifted(self, dx: float = 0.0, dy: float = 0.0) -> "Pose2D":
        return Pose2D(self.x + dx, self.y + dy)


@dataclass(frozen=True)
class Region:
    name: str
    center: Pose2D
    half_extents: tuple[float, float]

    def contains(self, pose: Pose2D) -> bool:
        return (
            abs(pose.x - self.center.x) <= self.half_extents[0]
            and abs(pose.y - self.center.y) <= self.half_extents[1]
        )


@dataclass(frozen=True)
class Obstacle:
    name: str
    center: Pose2D
    radius: float


@dataclass(frozen=True)
class MotionCandidate:
    label: str
    target_pose: Pose2D
    approach_pose: Pose2D
    retreat_pose: Pose2D


@dataclass
class TabletopScene:
    block_name: str
    block_pose: Pose2D
    start_region: Region
    goal_region: Region
    obstacle: Obstacle | None = None
    held_object: str | None = None

    def region(self, region_name: str) -> Region:
        if region_name == self.start_region.name:
            return self.start_region
        if region_name == self.goal_region.name:
            return self.goal_region
        raise KeyError(f"Unknown region: {region_name}")

    def pick_candidates(self, object_name: str) -> list[MotionCandidate]:
        if object_name != self.block_name:
            raise KeyError(f"Unknown object: {object_name}")
        base = self.block_pose
        return [
            MotionCandidate(
                label="pick-left-entry",
                target_pose=base,
                approach_pose=base.shifted(dx=-0.10),
                retreat_pose=base.shifted(dy=0.08),
            ),
            MotionCandidate(
                label="pick-top-entry",
                target_pose=base,
                approach_pose=base.shifted(dy=0.10),
                retreat_pose=base.shifted(dy=0.12),
            ),
        ]

    def place_candidates(self, region_name: str) -> list[MotionCandidate]:
        region = self.region(region_name)
        left_bias = region.center.shifted(dy=0.02)
        top_bias = region.center.shifted(dy=-0.02)
        return [
            MotionCandidate(
                label=f"place-left-entry@{region.name}",
                target_pose=left_bias,
                approach_pose=left_bias.shifted(dx=-0.11),
                retreat_pose=left_bias.shifted(dy=0.09),
            ),
            MotionCandidate(
                label=f"place-top-entry@{region.name}",
                target_pose=top_bias,
                approach_pose=top_bias.shifted(dy=0.11),
                retreat_pose=top_bias.shifted(dx=0.02, dy=0.12),
            ),
        ]

    def candidate_feasible(self, candidate: MotionCandidate, clearance: float = 0.045) -> bool:
        if self.obstacle is None:
            return True
        distance = _distance_point_to_segment(
            self.obstacle.center,
            candidate.approach_pose,
            candidate.target_pose,
        )
        return distance > (self.obstacle.radius + clearance)

    def commit_pick(self, object_name: str) -> None:
        if object_name != self.block_name:
            raise KeyError(f"Unknown object: {object_name}")
        self.held_object = object_name

    def commit_place(self, object_name: str, target_pose: Pose2D) -> None:
        if object_name != self.held_object:
            raise ValueError(
                f"Cannot place {object_name}; currently holding {self.held_object!r}"
            )
        self.block_pose = target_pose
        self.held_object = None

    def compact_summary(self) -> str:
        obstacle_summary = (
            f"{self.obstacle.name}@({self.obstacle.center.x:.2f}, {self.obstacle.center.y:.2f})"
            if self.obstacle is not None
            else "none"
        )
        return (
            f"block=({self.block_pose.x:.2f}, {self.block_pose.y:.2f}) | "
            f"holding={self.held_object or 'none'} | obstacle={obstacle_summary}"
        )

    def summary(self) -> str:
        return "\n".join(
            [
                "Tabletop scene:",
                f"  - block: {self.block_name} @ ({self.block_pose.x:.2f}, {self.block_pose.y:.2f})",
                f"  - start_region: center=({self.start_region.center.x:.2f}, {self.start_region.center.y:.2f})",
                f"  - goal_region: center=({self.goal_region.center.x:.2f}, {self.goal_region.center.y:.2f})",
                (
                    f"  - obstacle: {self.obstacle.name} @ ({self.obstacle.center.x:.2f}, {self.obstacle.center.y:.2f}), r={self.obstacle.radius:.2f}"
                    if self.obstacle is not None
                    else "  - obstacle: none"
                ),
            ]
        )


def build_demo_scene(*, include_obstacle: bool = True) -> TabletopScene:
    start_region = Region(
        name="start_region",
        center=Pose2D(0.22, -0.06),
        half_extents=(0.05, 0.05),
    )
    goal_region = Region(
        name="goal_region",
        center=Pose2D(0.45, 0.06),
        half_extents=(0.05, 0.05),
    )
    obstacle = (
        Obstacle(name="cup", center=Pose2D(0.34, 0.09), radius=0.05)
        if include_obstacle
        else None
    )
    return TabletopScene(
        block_name="red_block",
        block_pose=start_region.center,
        start_region=start_region,
        goal_region=goal_region,
        obstacle=obstacle,
    )


def _distance_point_to_segment(point: Pose2D, start: Pose2D, end: Pose2D) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    if dx == 0.0 and dy == 0.0:
        return hypot(point.x - start.x, point.y - start.y)

    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = start.x + t * dx
    proj_y = start.y + t * dy
    return hypot(point.x - proj_x, point.y - proj_y)
