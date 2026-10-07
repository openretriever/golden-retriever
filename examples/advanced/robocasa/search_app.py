"""Run one multi-drawer search episode and print its trace.

Mock by default: needs no simulator, no assets, no GPU and no network.

    pixi run demo-robocasa-search-mock

Imports nothing heavier than the Retriever runtime, so the lane stays
import-safe with no simulator installed; a real RoboCasa backend arrives in
Phase 3 behind the same `SearchWorld` boundary.
"""

from __future__ import annotations

import argparse

from .search_belief import SearchBelief
from .search_plan import plan_ahead
from .search_policy import get_policy
from .search_scenes import SCENES, get_scene
from .search_trace import SearchTrace, TraceRecord
from .search_world import MockSearchWorld, SearchWorld, search_status

__all__ = ["run_episode", "main"]

# A search can never need more looks than there are containers; anything beyond
# that means a policy is cycling, and the trace should show that rather than
# hang.
_STEP_SAFETY_FACTOR = 2


def run_episode(
    *,
    scene_id: str = "three_drawer_stack",
    seed: int = 0,
    policy_name: str = "nearest-first",
    world: SearchWorld | None = None,
    scene=None,
    max_steps: int | None = None,
    close_world: bool = True,
) -> SearchTrace:
    """Search until the target is found, the scene is exhausted, or we cap out.

    `scene` overrides `scene_id` — a real backend derives its scene from the
    live kitchen, so the containers are actual fixtures rather than names from
    the registry. `close_world=False` keeps a simulator alive across episodes,
    which matters because building a kitchen costs far more than a search does.
    """
    scene = scene if scene is not None else get_scene(scene_id)
    policy = get_policy(policy_name)
    world = world if world is not None else MockSearchWorld()
    limit = max_steps or len(scene.containers) * _STEP_SAFETY_FACTOR

    world.reset(scene, seed)
    belief = SearchBelief.uniform(scene.container_ids)
    trace = SearchTrace(
        scene_id=scene.scene_id, seed=seed, policy=policy.name, target=scene.target
    )
    total = len(scene.containers)
    trace.append(
        TraceRecord(
            step=0,
            event="episode_started",
            reason=f"uniform over {total} containers",
            belief_after=belief.as_dict(),
            progress=0.0,
            status="IN_PROGRESS",
        )
    )

    def status_of(inspected: int, found: bool, error: str | None = None):
        """Prefer the world's own report; fall back for minimal test doubles."""
        reporter = getattr(world, "status", None)
        if reporter is not None and error is None:
            try:
                return reporter()
            except Exception:  # noqa: BLE001 - fall back rather than lose the trace
                pass
        return search_status(
            inspected=inspected, total=total, found=found, error_message=error
        )

    seen: set[str] = set()
    position = scene.robot_home
    step = 0

    plan = plan_ahead(policy, belief, scene, position, revision=0)
    trace.append(
        TraceRecord(
            step=0,
            event="planned",
            plan=plan.as_dict(),
            reason=plan.basis,
            progress=0.0,
            status="IN_PROGRESS",
        )
    )
    revision = 0

    try:
        while step < limit:
            decision = policy.decide(belief, scene, position)

            # A plan is a prediction. Acting on it moves the robot, and
            # `nearest-first` measures cost from wherever it now stands, so the
            # remaining order can legitimately change. Record that as a replan
            # rather than silently diverging from the plan on file.
            #
            # Only while the search is still live: once the belief is decided
            # there is nothing left to plan, and planning anyway would log an
            # empty "replan" on every successful episode.
            if not decision.stops and step > 0:
                expected = plan.container_ids[1:]
                current = plan_ahead(
                    policy, belief, scene, position, revision=revision
                )
                if current.container_ids != expected:
                    revision += 1
                    plan = plan_ahead(
                        policy, belief, scene, position, revision=revision
                    )
                    trace.append(
                        TraceRecord(
                            step=step,
                            event="planned",
                            plan=plan.as_dict(),
                            reason="remaining order changed after the last observation",
                            progress=float(
                                status_of(len(seen), belief.found).progress or 0.0
                            ),
                            status="IN_PROGRESS",
                        )
                    )
                else:
                    plan = current

            if decision.stops:
                outcome = status_of(len(seen), belief.found)
                trace.append(
                    TraceRecord(
                        step=step,
                        event="terminated",
                        reason=decision.reason,
                        progress=float(outcome.progress or 0.0),
                        status=outcome.status,
                        error_message=outcome.error_message or "",
                    )
                )
                return trace

            step += 1
            before = belief.as_dict()
            leg = scene.travel_cost(decision.container_id, position)
            try:
                result = world.inspect(decision.container_id)
            except Exception as exc:  # noqa: BLE001 - a failure must be inspectable
                failure = status_of(len(seen), False, f"{type(exc).__name__}: {exc}")
                trace.append(
                    TraceRecord(
                        step=step,
                        event="failed",
                        container_id=decision.container_id,
                        reason=decision.reason,
                        belief_before=before,
                        progress=float(failure.progress or 0.0),
                        status=failure.status,
                        error_message=failure.error_message or "",
                    )
                )
                return trace

            position = scene.container(decision.container_id).position
            repeated = decision.container_id in seen
            seen.add(decision.container_id)
            belief = belief.update(result)
            progress = status_of(len(seen), belief.found)
            trace.append(
                TraceRecord(
                    step=step,
                    event="inspected",
                    container_id=decision.container_id,
                    found=result.found,
                    reason=decision.reason,
                    belief_before=before,
                    belief_after=belief.as_dict(),
                    repeated=repeated,
                    travel_cost=leg,
                    progress=float(progress.progress or 0.0),
                    status=progress.status,
                )
            )

        exceeded = search_status(
            inspected=len(seen),
            total=total,
            found=belief.found,
            error_message=f"step limit of {limit} reached without a decision to stop",
        )
        trace.append(
            TraceRecord(
                step=step,
                event="failed",
                reason="step limit reached",
                progress=float(exceeded.progress or 0.0),
                status=exceeded.status,
                error_message=exceeded.error_message or "",
            )
        )
        return trace
    finally:
        if close_world:
            world.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="three_drawer_stack", choices=sorted(SCENES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy", default="nearest-first", choices=["fixed-order", "nearest-first"]
    )
    parser.add_argument(
        "--trace", default="", help="write the JSONL trace to this path as well"
    )
    parser.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "robocasa"],
        help="mock needs nothing installed; robocasa needs the simulator and assets",
    )
    parser.add_argument(
        "--layout", type=int, default=1, help="RoboCasa kitchen layout id (robocasa backend)"
    )
    parser.add_argument(
        "--containers", type=int, default=4, help="how many drawers to search (robocasa backend)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.backend == "robocasa":
        from .search_robocasa_world import RoboCasaSearchWorld

        world = RoboCasaSearchWorld(layout_id=args.layout, max_containers=args.containers)
        scene = world.build_scene(scene_id=f"robocasa_layout{args.layout}")
        trace = run_episode(
            scene=scene, seed=args.seed, policy_name=args.policy, world=world
        )
    else:
        trace = run_episode(
            scene_id=args.scene, seed=args.seed, policy_name=args.policy
        )
    print(trace.render())
    if args.trace:
        print(f"\nwrote {trace.write(args.trace)}")
    return 0 if trace.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
