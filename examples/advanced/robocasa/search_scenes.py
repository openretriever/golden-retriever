"""Search scenes as plain data.

Dependency-free on purpose — no MuJoCo, no RoboCasa, no NumPy, not even the
Retriever runtime — so the mock lane, the tests and the docs all describe the
same scenes without a simulator or an asset pack present. `world.py` binds
these to a backend; nothing here knows how a container is opened.

Where the target hides is a pure function of `(scene_id, seed)`, computed here
rather than delegated to simulator RNG. That is what lets the mock world and a
real RoboCasa kitchen agree on the answer, and what makes "the target location
changes across seeds" checkable without starting a simulator.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import dist

__all__ = [
    "Container",
    "SearchScene",
    "SCENES",
    "get_scene",
]


@dataclass(frozen=True)
class Container:
    """One searchable container: a drawer or a cabinet."""

    container_id: str
    label: str
    kind: str = "drawer"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SearchScene:
    """A fixed scene: some containers, one target, one robot start pose."""

    scene_id: str
    target: str
    containers: tuple[Container, ...]
    robot_home: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.containers:
            raise ValueError(f"scene {self.scene_id!r} has no containers")
        ids = [c.container_id for c in self.containers]
        if len(set(ids)) != len(ids):
            raise ValueError(f"scene {self.scene_id!r} has duplicate container ids")

    @property
    def container_ids(self) -> tuple[str, ...]:
        return tuple(c.container_id for c in self.containers)

    def container(self, container_id: str) -> Container:
        for candidate in self.containers:
            if candidate.container_id == container_id:
                return candidate
        raise KeyError(f"{container_id!r} is not a container in {self.scene_id!r}")

    def target_container(self, seed: int) -> str:
        """Which container holds the target for this seed.

        SHA-256 rather than `hash()` or `random`: the placement has to be
        identical across processes, platforms and Python versions, and
        `hash()` on a str is salted per process.
        """
        digest = hashlib.sha256(f"{self.scene_id}:{seed}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(self.containers)
        return self.containers[index].container_id

    def travel_cost(self, container_id: str, origin: tuple[float, float, float] | None = None) -> float:
        """Straight-line distance from `origin` (default: robot home)."""
        return dist(origin if origin is not None else self.robot_home,
                    self.container(container_id).position)


# --- scene registry -------------------------------------------------------
# Three fixed scenes, chosen so they measure different things rather than
# being three variations of one layout. Declaration order is deliberately not
# distance order anywhere: a scene author has no reason to list containers
# nearest-first, and if they did, the two policies would collapse into one.

_THREE_DRAWER_STACK = SearchScene(
    scene_id="three_drawer_stack",
    target="pepper_shaker",
    robot_home=(0.0, 0.0, 0.0),
    containers=(
        Container("drawer_top", "Top drawer", "drawer", (0.60, 0.00, 0.85)),
        Container("drawer_middle", "Middle drawer", "drawer", (0.60, 0.00, 0.60)),
        Container("drawer_bottom", "Bottom drawer", "drawer", (0.60, 0.00, 0.35)),
        Container("cabinet_left", "Left cabinet", "cabinet", (0.95, -0.70, 0.55)),
    ),
)

# A galley run: containers spread down a long counter, so travel cost varies by
# almost 3x from end to end. This is where ordering by cost should pay off most.
_GALLEY_RUN = SearchScene(
    scene_id="galley_run",
    target="pepper_shaker",
    robot_home=(0.0, 0.0, 0.0),
    containers=(
        Container("cabinet_far", "Far upper cabinet", "cabinet", (2.35, -0.25, 1.45)),
        Container("drawer_mid", "Middle drawer", "drawer", (1.40, -0.25, 0.80)),
        Container("cabinet_near", "Near upper cabinet", "cabinet", (0.60, -0.25, 1.45)),
        Container("drawer_far", "Far drawer", "drawer", (2.30, -0.25, 0.80)),
        Container("drawer_near", "Near drawer", "drawer", (0.55, -0.25, 0.80)),
        Container("cabinet_mid", "Middle upper cabinet", "cabinet", (1.45, -0.25, 1.45)),
    ),
)

# An island ring: every container is exactly the same distance away, so cost
# ordering has nothing to work with. The control case — a policy that claims an
# advantage here is reporting noise, not skill.
_ISLAND_RING = SearchScene(
    scene_id="island_ring",
    target="pepper_shaker",
    robot_home=(0.0, 0.0, 0.0),
    containers=(
        Container("island_north", "North island drawer", "drawer", (0.00, 1.00, 0.70)),
        Container("island_east", "East island cabinet", "cabinet", (1.00, 0.00, 0.70)),
        Container("island_south", "South island drawer", "drawer", (0.00, -1.00, 0.70)),
        Container("island_west", "West island cabinet", "cabinet", (-1.00, 0.00, 0.70)),
    ),
)

SCENES: dict[str, SearchScene] = {
    scene.scene_id: scene
    for scene in (_THREE_DRAWER_STACK, _GALLEY_RUN, _ISLAND_RING)
}


def get_scene(scene_id: str) -> SearchScene:
    try:
        return SCENES[scene_id]
    except KeyError:
        known = ", ".join(sorted(SCENES)) or "(none)"
        raise KeyError(f"unknown scene {scene_id!r}; known scenes: {known}") from None
