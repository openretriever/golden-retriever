"""D1 Phase 0 spike — can RoboCasa do what multi-drawer search needs?

Three questions, answered against the real installed RoboCasa:

  Q1. Can we enumerate a kitchen's containers (cabinets / drawers) by name?
  Q2. Can we read back whether a named container is open or closed?
  Q3. Can we place a chosen object inside a *named* container?

Q1+Q2 are what a search agent needs to act and observe. Q3 is what lets a
target be hidden in a seed-chosen container. If Q3 fails, D1's scenes have to
come from somewhere other than RoboCasa's own kitchens.

Run:  pixi run --locked -e robocasa python <this file>
"""

from __future__ import annotations

import inspect
import sys
import traceback

VERDICT: dict[str, str] = {}


def head(n: int, text: str) -> None:
    print(f"\n{'=' * 72}\nQ{n}. {text}\n{'=' * 72}")


def probe_fixture_classes() -> None:
    """What container fixtures exist, and what state API do they expose?"""
    head(0, "What container fixture classes does RoboCasa ship?")
    try:
        from robocasa.models.fixtures import cabinets
    except Exception:
        traceback.print_exc()
        VERDICT["fixture_classes"] = "FAIL - could not import robocasa.models.fixtures.cabinets"
        return

    names = [
        n for n, o in inspect.getmembers(cabinets, inspect.isclass)
        if o.__module__.startswith("robocasa")
    ]
    print(f"cabinet/drawer classes: {names}")

    # The state API is the thing that matters: can we read and set open-ness?
    for cls_name in names:
        cls = getattr(cabinets, cls_name)
        state_api = [
            m for m in dir(cls)
            if any(k in m.lower() for k in ("door_state", "open", "close", "joint"))
            and not m.startswith("__")
        ]
        if state_api:
            print(f"  {cls_name}: {sorted(state_api)}")
    VERDICT["fixture_classes"] = f"OK - {len(names)} classes"


CANDIDATE_TASKS = (
    "PickPlaceCounterToDrawer",  # object actually goes INTO a drawer
    "OpenDrawer",
    "OpenCabinet",
)


def build_env(tasks: tuple[str, ...] = CANDIDATE_TASKS):
    """Build one RoboCasa kitchen, trying task names until one works."""
    import robocasa  # noqa: F401
    import robosuite

    last_exc = None
    for task in tasks:
        print(f"trying robosuite.make(env_name={task!r}) ...")
        try:
            env = robosuite.make(
                env_name=task,
                robots="PandaOmron",
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
            )
            env.reset()
            print(f"  built {task!r}")
            VERDICT["env_task"] = task
            return env
        except Exception as exc:  # noqa: BLE001 - spike: report and keep trying
            last_exc = exc
            msg = str(exc)
            if len(msg) > 200:  # the "not registered" error dumps ~400 env names
                msg = msg[:200] + " ...[truncated]"
            print(f"  {type(exc).__name__}: {msg}")
    raise RuntimeError(f"no candidate task built; last error: {last_exc}")


def q1_enumerate(env) -> list:
    head(1, "Can we enumerate containers by name?")
    fixtures = getattr(env, "fixtures", None)
    if fixtures is None:
        VERDICT["q1_enumerate"] = "FAIL - env has no .fixtures"
        print("env has no .fixtures attribute")
        return []

    print(f"env.fixtures is a {type(fixtures).__name__} with {len(fixtures)} entries")
    containers = []
    items = fixtures.items() if hasattr(fixtures, "items") else enumerate(fixtures)
    for name, fx in items:
        kind = type(fx).__name__
        if any(k in kind.lower() for k in ("cabinet", "drawer")):
            containers.append((name, fx, kind))

    print(f"\ncontainer-like fixtures ({len(containers)}):")
    for name, _fx, kind in containers:
        print(f"  {name:<40} {kind}")

    VERDICT["q1_enumerate"] = (
        f"OK - {len(containers)} containers by name" if containers
        else "FAIL - no cabinet/drawer fixtures found"
    )
    return containers


def q2_readback(env, containers: list) -> None:
    head(2, "Can we read back open/closed state per container?")
    if not containers:
        VERDICT["q2_readback"] = "SKIP - no containers"
        return

    worked = 0
    for name, fx, kind in containers[:8]:
        for meth in ("get_door_state",):
            fn = getattr(fx, meth, None)
            if fn is None:
                continue
            try:
                state = fn(env)
                print(f"  {name:<40} {meth}(env) -> {state}")
                worked += 1
            except Exception as exc:
                print(f"  {name:<40} {meth}(env) raised {type(exc).__name__}: {exc}")
            break
        else:
            print(f"  {name:<40} no get_door_state (type {kind})")

    VERDICT["q2_readback"] = (
        f"OK - {worked}/{min(len(containers), 8)} containers report state"
        if worked else "FAIL - no container exposed readable state"
    )


def q3_placement(env, containers: list) -> None:
    head(3, "Can an object be placed inside a NAMED container?")

    # How does this env declare its object placements? That config schema is
    # the thing D1 would have to synthesize per seed.
    fn = getattr(env, "_get_obj_cfgs", None)
    if fn is None:
        print("env has no _get_obj_cfgs; cannot inspect placement schema")
        VERDICT["q3_placement"] = "FAIL - no _get_obj_cfgs"
        return

    try:
        cfgs = fn()
    except Exception:
        traceback.print_exc()
        VERDICT["q3_placement"] = "FAIL - _get_obj_cfgs raised"
        return

    print(f"_get_obj_cfgs() returned {len(cfgs)} object config(s). Schema of each:\n")
    fixture_refs = 0
    for cfg in cfgs:
        name = cfg.get("name", "?")
        placement = cfg.get("placement", {}) or {}
        print(f"  object {name!r}")
        print(f"    keys: {sorted(cfg.keys())}")
        print(f"    placement keys: {sorted(placement.keys())}")
        if "fixture" in placement:
            fixture_refs += 1
            fx = placement["fixture"]
            print(f"    placement['fixture'] = {fx!r} (type {type(fx).__name__})")
        for k in ("sample_region_kwargs", "size", "pos", "offset", "try_to_place_in"):
            if k in placement:
                print(f"    placement[{k!r}] = {placement[k]!r}")
        print()

    if fixture_refs:
        VERDICT["q3_placement"] = (
            f"OK - {fixture_refs} object cfg(s) target a fixture directly; "
            "placement['fixture'] is the seam for seed-chosen hiding"
        )
    else:
        VERDICT["q3_placement"] = (
            "PARTIAL - no cfg targets a fixture in this task; check a "
            "pick-from-cabinet task before concluding"
        )


def main() -> int:
    print("D1 Phase 0 — RoboCasa fixture spike")
    probe_fixture_classes()

    env = None
    try:
        env = build_env()
    except Exception:
        traceback.print_exc()
        VERDICT["env_build"] = "FAIL - could not build kitchen env"
    else:
        VERDICT["env_build"] = "OK"
        try:
            containers = q1_enumerate(env)
            q2_readback(env, containers)
            q3_placement(env, containers)
        finally:
            try:
                env.close()
            except Exception:
                pass

    print(f"\n{'=' * 72}\nVERDICT\n{'=' * 72}")
    for k, v in VERDICT.items():
        print(f"  {k:<20} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
