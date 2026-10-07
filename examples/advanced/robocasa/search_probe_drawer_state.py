"""Phase 0 spike: can a RoboCasa drawer's open/closed state be read and driven?

Answers the question D1's "inspect a container" primitive depends on. Run:

    pixi run --locked -e robocasa python -m \
        examples.advanced.robocasa.search_probe_drawer_state
"""

from __future__ import annotations

import robocasa  # noqa: F401  (registers the kitchen environments)
import robosuite
from robocasa.models.fixtures.cabinets import Drawer


def main() -> int:
    env = robosuite.make(
        env_name="PickPlaceDrawerToCounter",
        robots="PandaOmron",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        layout_ids=[1],
        style_ids=[1],
        seed=7,
    )
    try:
        env.reset()
        drawer = env.drawer
        print(f"target drawer: {drawer.name}")

        def report(label: str) -> None:
            state = {k: round(float(v), 3) for k, v in drawer.get_door_state(env).items()}
            print(
                f"  {label:<18} state={state} "
                f"open={drawer.is_open(env)} closed={drawer.is_closed(env)}"
            )

        report("initial:")
        drawer.close_door(env)
        env.sim.forward()
        report("after close_door:")

        # partial_open=True settles near 0.868, below is_open()'s 0.90 threshold.
        drawer.open_door(env)
        env.sim.forward()
        report("after open_door:")

        drawer.set_door_state(min=0.0, max=0.0, env=env)
        env.sim.forward()
        report("after set(0,0):")

        drawers = [(n, f) for n, f in env.fixtures.items() if isinstance(f, Drawer)]
        print(f"\nindependently addressable drawers in this kitchen: {len(drawers)}")
        for _name, fixture in drawers[:5]:
            fixture.close_door(env)
        env.sim.forward()
        closed = [f"{n}:{f.is_closed(env)}" for n, f in drawers[:5]]
        print(f"  closed first 5 -> {closed}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
