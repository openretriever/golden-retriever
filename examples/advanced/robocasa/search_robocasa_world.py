"""A `SearchWorld` backed by a real RoboCasa kitchen.

Same boundary as `MockSearchWorld`, so the belief, the policies and the trace
run unchanged against real physics. RoboCasa is imported lazily inside the
methods, which keeps the lane import-safe with no simulator installed and keeps
`pixi run test` green in CI.

Two things make this reproducible, both established by the Phase 0 spike:

* Pinning `layout_ids`, `style_ids` and RoboCasa's own `seed` makes the kitchen
  and its container names identical across resets *and* across fresh env
  construction. Without all three, RoboCasa reshuffles the kitchen every reset.
* Which container hides the target stays a pure function of `(scene_id, seed)`
  computed in `scenes.py`, never RoboCasa's RNG, so the mock and real backends
  agree on the answer.

Scope: opening a container drives its joint directly through RoboCasa's fixture
API rather than grasping the handle. D1 measures search efficiency — which
containers get opened, in what order, and how the belief moves — and the
catalog assigns contact-level skill execution to D4. `examples/advanced/
robocasa_drawer/` is the reference for what real contact-based opening costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .search_belief import InspectionResult
from .search_scenes import Container, SearchScene
from .search_world import SearchObservation, SearchVerification, search_status

__all__ = ["RoboCasaSearchWorld"]

# RoboCasa filters reset regions by height and defaults to (0.45, 1.50), which
# silently drops drawers in the lower half of a stack. We want all of them.
_FULL_Z_RANGE = (0.0, 3.0)


@dataclass
class RoboCasaSearchWorld:
    """Search containers in a real RoboCasa kitchen."""

    env_name: str = "PickPlaceCounterToDrawer"
    layout_id: int = 1
    style_id: int = 1
    robocasa_seed: int = 7
    max_containers: int = 4
    target_object: str = "obj"
    settle_steps: int = 40
    containment_pad: float = 0.03

    env: object | None = None
    scene: SearchScene | None = None
    _drawers: dict = field(default_factory=dict)
    _target_container: str = ""
    _opened: list = field(default_factory=list)
    _step: int = 0
    _found: bool = False

    # --- construction -----------------------------------------------------

    def _ensure_env(self):
        if self.env is not None:
            return self.env
        import robocasa  # noqa: F401  (registers the kitchen environments)
        import robosuite

        self.env = robosuite.make(
            env_name=self.env_name,
            robots="PandaOmron",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            layout_ids=[self.layout_id],
            style_ids=[self.style_id],
            seed=self.robocasa_seed,
        )
        return self.env

    def _collect_drawers(self) -> dict:
        from robocasa.models.fixtures.cabinets import Drawer

        # Sorted by name so the mapping from scene container to real fixture is
        # stable; dict order out of RoboCasa is not guaranteed to be.
        return {
            name: fixture
            for name, fixture in sorted(self.env.fixtures.items())
            if isinstance(fixture, Drawer)
        }

    def build_scene(self, scene_id: str = "robocasa_kitchen", target: str = "obj") -> SearchScene:
        """Derive a `SearchScene` from the live kitchen.

        Container ids, positions and count all come from the real fixtures, so
        travel costs are real distances rather than invented ones.
        """
        import numpy as np

        env = self._ensure_env()
        env.reset()
        drawers = self._collect_drawers()
        if not drawers:
            raise RuntimeError(
                f"{self.env_name} layout={self.layout_id} exposes no drawers"
            )

        home = self._robot_home()

        def distance(name: str) -> float:
            return float(
                np.linalg.norm(np.asarray(drawers[name].pos) - np.asarray(home))
            )

        # One drawer per stack, nearest stacks first.
        #
        # RoboCasa groups drawers into vertical stacks: members of a stack share
        # x and y and differ only in height, so they sit within ~0.2 m of each
        # other while stacks are metres apart. Taking the nearest N by distance
        # therefore grabs a single stack, every travel cost lands within a few
        # centimetres, and the policy comparison collapses into noise. Spreading
        # the search across stacks is what makes the geometry mean anything.
        # Ties break by name so the selection stays deterministic.
        by_stack: dict[str, list[str]] = {}
        for name in drawers:
            by_stack.setdefault(name.rsplit("_", 1)[0], []).append(name)

        representatives = sorted(
            (min(members, key=lambda n: (distance(n), n)) for members in by_stack.values()),
            key=lambda n: (distance(n), n),
        )
        chosen = representatives[: self.max_containers]
        if len(chosen) < self.max_containers:  # fewer stacks than asked for
            remainder = sorted(
                (n for n in drawers if n not in chosen), key=lambda n: (distance(n), n)
            )
            chosen += remainder[: self.max_containers - len(chosen)]

        # Selection is by distance, but declaration order must not be: emitting
        # them nearest-first would make `fixed-order` silently identical to
        # `nearest-first` and the comparison would measure nothing. Name order
        # is arbitrary with respect to geometry and stays deterministic.
        containers = tuple(
            Container(
                container_id=name,
                label=name.replace("_", " "),
                kind="drawer",
                position=tuple(float(v) for v in np.asarray(drawers[name].pos)),
            )
            for name in sorted(chosen)
        )
        return SearchScene(
            scene_id=scene_id,
            target=target,
            containers=containers,
            robot_home=self._robot_home(),
        )

    def _robot_home(self) -> tuple[float, float, float]:
        """Where the robot actually stands.

        Order matters. On a mobile-base robot `robot0_base` is a parked dummy
        body sitting at (10, 10, 0) while the arm really hangs off
        `mobilebase0_base` — reading the wrong one puts the robot 14 m from its
        own kitchen and makes every travel cost meaningless.
        """
        import numpy as np

        for body in ("mobilebase0_base", "robot0_link0", "robot0_base"):
            try:
                position = np.asarray(self.env.sim.data.get_body_xpos(body))
            except Exception:  # noqa: BLE001 - probing for whichever body exists
                continue
            if np.all(np.abs(position) < 9.0):  # reject the parked dummy
                return tuple(float(v) for v in position)
        return (0.0, 0.0, 0.0)

    # --- geometry ---------------------------------------------------------

    def _interior(self, fixture):
        import numpy as np

        region = fixture.get_reset_regions(self.env, z_range=_FULL_Z_RANGE)["int"]
        centre = np.asarray(fixture.pos) + np.asarray(region["offset"])
        return centre, np.asarray(region["size"]), float(region["height"])

    def _object_position(self):
        import numpy as np

        return np.asarray(self.env.sim.data.body_xpos[self.env.obj_body_id[self.target_object]])

    def _teleport_target(self, xyz) -> None:
        joint = self.env.objects[self.target_object].joints[0]
        address = self.env.sim.model.get_joint_qpos_addr(joint)
        low = address[0] if isinstance(address, tuple) else address
        self.env.sim.data.qpos[low : low + 3] = xyz
        self.env.sim.data.qpos[low + 3 : low + 7] = [1.0, 0.0, 0.0, 0.0]
        self.env.sim.data.qvel[:] = 0.0
        self.env.sim.forward()

    def _contains(self, fixture, position) -> bool:
        centre, size, height = self._interior(fixture)
        pad = self.containment_pad
        return bool(
            abs(position[0] - centre[0]) <= size[0] / 2 + pad
            and abs(position[1] - centre[1]) <= size[1] / 2 + pad
            and -0.10 <= position[2] - centre[2] <= height + 0.10
        )

    def _settle(self, steps: int | None = None) -> None:
        for _ in range(steps if steps is not None else self.settle_steps):
            self.env.sim.step()

    # --- SearchWorld ------------------------------------------------------

    def reset(self, scene: SearchScene, seed: int) -> SearchObservation:
        import numpy as np

        env = self._ensure_env()
        env.reset()
        self._drawers = self._collect_drawers()

        missing = [c for c in scene.container_ids if c not in self._drawers]
        if missing:
            raise KeyError(
                f"scene {scene.scene_id!r} names containers this kitchen does not "
                f"have: {missing}. Build the scene with build_scene()."
            )

        self.scene = scene
        self._opened = []
        self._step = 0
        self._found = False
        self._target_container = scene.target_container(seed)

        # Start from a closed kitchen so "open" means something.
        for fixture in self._drawers.values():
            fixture.close_door(env)
        env.sim.forward()

        centre, _size, _height = self._interior(self._drawers[self._target_container])
        self._teleport_target(np.asarray(centre) + np.array([0.0, 0.0, 0.04]))
        self._settle()
        return self.observe()

    def inspect(self, container_id: str) -> InspectionResult:
        if self.scene is None:
            raise RuntimeError("reset() the world before inspecting it")
        fixture = self._drawers[container_id]
        self._step += 1
        if container_id not in self._opened:
            self._opened.append(container_id)

        # partial_open=True settles around 0.868 and never satisfies is_open()'s
        # 0.90 threshold, so drive the joint fully open explicitly.
        fixture.set_door_state(min=1.0, max=1.0, env=self.env)
        self.env.sim.forward()
        self._settle()

        found = self._contains(fixture, self._object_position())
        self._found = self._found or found
        return InspectionResult(container_id=container_id, found=found, step=self._step)

    def observe(self) -> SearchObservation:
        return SearchObservation(
            scene_id=self.scene.scene_id if self.scene else "",
            opened=tuple(self._opened),
            holding_target=self._found,
            step=self._step,
        )

    def status(self):
        if self.scene is None:
            raise RuntimeError("reset() the world before asking for its status")
        return search_status(
            inspected=len(self._opened),
            total=len(self.scene.containers),
            found=self._found,
        )

    def verify(self) -> SearchVerification:
        """RoboCasa's geometry, not the agent's belief, decides this."""
        actually_in = ""
        if self.scene is not None:
            position = self._object_position()
            for name, fixture in self._drawers.items():
                if self._contains(fixture, position):
                    actually_in = name
                    break
        return SearchVerification(
            success=self._found,
            target_container=self._target_container,
            message=(
                f"target hidden in {self._target_container}, "
                f"geometry reports it in {actually_in or 'no container'}"
            ),
        )

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None
