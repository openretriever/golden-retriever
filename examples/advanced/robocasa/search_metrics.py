"""The D1 metrics, computed from traces rather than from internals.

The catalog's done-definition requires the search to be inspectable "without
reading internal simulator code", so every number here is derived from a trace
— and `EpisodeMetrics.from_jsonl` reads one back from disk, which is the proof
that a trace file is self-sufficient. Nothing in this module imports a
simulator, a policy or a belief.

One deliberate omission: elapsed wall-clock time is *not* written into the
trace. It would change between runs and break the byte-identical replay check
that the determinism test depends on. Timing is measured here, around the
episode, where it cannot contaminate the artifact.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

__all__ = [
    "EpisodeMetrics",
    "AggregateMetrics",
    "replay_is_deterministic",
    "sweep",
    "render_table",
]

_MASS = 1e-9


@dataclass(frozen=True)
class EpisodeMetrics:
    """The catalog's metrics for one episode."""

    scene_id: str
    seed: int
    policy: str
    success: bool
    containers_opened: int
    repeated_inspections: int
    unnecessary_inspections: int
    steps: int
    travel_cost: float
    elapsed_seconds: float = 0.0
    status: str = "IN_PROGRESS"
    progress: float = 0.0
    error_message: str = ""
    replans: int = 0
    followed_initial_plan: bool = True

    @property
    def failed(self) -> bool:
        return self.status == "FAILURE"

    @classmethod
    def from_trace(cls, trace, elapsed_seconds: float = 0.0) -> EpisodeMetrics:
        return cls(
            status=trace.status,
            progress=trace.final_progress,
            error_message=trace.error_message,
            replans=trace.replans,
            followed_initial_plan=trace.followed_initial_plan,
            scene_id=trace.scene_id,
            seed=trace.seed,
            policy=trace.policy,
            success=trace.succeeded,
            containers_opened=trace.containers_opened,
            repeated_inspections=trace.repeated_inspections,
            unnecessary_inspections=_unnecessary(
                [
                    (r.belief_before, r.found)
                    for r in trace.inspections
                ]
            ),
            steps=trace.steps,
            travel_cost=trace.total_travel_cost,
            elapsed_seconds=elapsed_seconds,
        )

    @classmethod
    def from_jsonl(cls, source: str | Path) -> EpisodeMetrics:
        """Recompute everything from a written trace file.

        This is the audit path: if these numbers match the in-memory ones, the
        trace really does carry the whole story.
        """
        text = Path(source).read_text(encoding="utf-8") if _looks_like_path(source) else str(source)
        lines = [line for line in text.strip().splitlines() if line.strip()]
        header = json.loads(lines[0])
        records = [json.loads(line) for line in lines[1:]]
        inspections = [r for r in records if r.get("event") == "inspected"]
        plans = [r for r in records if r.get("event") == "planned"]
        opened = [r["container_id"] for r in inspections]
        initial = list(plans[0].get("plan", {}).get("container_ids", [])) if plans else []
        terminal = next(
            (r for r in reversed(records) if r.get("status")),
            {},
        )

        return cls(
            status=terminal.get("status", "IN_PROGRESS"),
            progress=float(records[-1].get("progress", 0.0)) if records else 0.0,
            error_message=next(
                (r.get("error_message", "") for r in reversed(records) if r.get("error_message")),
                "",
            ),
            replans=max(0, len(plans) - 1),
            followed_initial_plan=opened == initial[: len(opened)],
            scene_id=header["scene_id"],
            seed=header["seed"],
            policy=header["policy"],
            success=any(r.get("found") for r in inspections),
            containers_opened=len({r["container_id"] for r in inspections}),
            repeated_inspections=sum(1 for r in inspections if r.get("repeated")),
            unnecessary_inspections=_unnecessary(
                [(r.get("belief_before", {}), r.get("found", False)) for r in inspections]
            ),
            steps=len(inspections),
            travel_cost=sum(float(r.get("travel_cost", 0.0)) for r in inspections),
        )


def _looks_like_path(source) -> bool:
    return isinstance(source, Path) or ("\n" not in str(source) and len(str(source)) < 4096)


def _unnecessary(inspections) -> int:
    """Inspections whose outcome was already certain from the belief.

    Opening the only container that can still hold the target yields no
    information — the answer was already known by elimination.

    Read this as a property of the *stopping rule*, not of the ordering policy.
    Termination currently requires physically finding the target, so when the
    target happens to be in the last-searched container the agent must open it
    even though the belief has collapsed. That costs one such inspection with
    probability 1/n, which is why every policy scores about the same here
    (~0.2 across a 4-to-6 container scene) rather than the metric separating
    them.

    It is still worth recording: an agent that inferred "it must be in the last
    one" and terminated without opening would drive this to zero, and that is a
    real behaviour D3 and D6 may want to compare against.
    """
    count = 0
    for belief_before, _found in inspections:
        candidates = [p for p in belief_before.values() if p > _MASS]
        if len(candidates) == 1:
            count += 1
    return count


@dataclass(frozen=True)
class AggregateMetrics:
    """One row of the results table."""

    scene_id: str
    policy: str
    episodes: int
    success_rate: float
    failure_rate: float
    mean_progress: float
    mean_containers_opened: float
    mean_repeated: float
    mean_unnecessary: float
    mean_steps: float
    mean_travel: float
    mean_replans: float
    mean_elapsed_ms: float

    @classmethod
    def over(cls, episodes: list[EpisodeMetrics]) -> AggregateMetrics:
        if not episodes:
            raise ValueError("cannot aggregate an empty set of episodes")
        first = episodes[0]
        return cls(
            scene_id=first.scene_id,
            policy=first.policy,
            episodes=len(episodes),
            success_rate=mean(1.0 if e.success else 0.0 for e in episodes),
            failure_rate=mean(1.0 if e.failed else 0.0 for e in episodes),
            mean_progress=mean(e.progress for e in episodes),
            mean_containers_opened=mean(e.containers_opened for e in episodes),
            mean_repeated=mean(e.repeated_inspections for e in episodes),
            mean_unnecessary=mean(e.unnecessary_inspections for e in episodes),
            mean_steps=mean(e.steps for e in episodes),
            mean_travel=mean(e.travel_cost for e in episodes),
            mean_replans=mean(e.replans for e in episodes),
            mean_elapsed_ms=mean(e.elapsed_seconds for e in episodes) * 1000.0,
        )


def replay_is_deterministic(
    *, scene_id: str, seed: int, policy_name: str, runs: int = 2
) -> bool:
    """Run the same episode `runs` times and compare the traces byte for byte.

    This is the check that discharges the catalog's "the same fixed scenes can
    be reset and replayed".
    """
    from .search_app import run_episode

    rendered = {
        run_episode(scene_id=scene_id, seed=seed, policy_name=policy_name).to_jsonl()
        for _ in range(runs)
    }
    return len(rendered) == 1


def sweep(
    *,
    scene_ids: list[str],
    policy_names: list[str],
    seeds: range | list[int],
    scene=None,
    world=None,
) -> list[EpisodeMetrics]:
    """Run every (scene, policy, seed) combination and collect its metrics."""
    from .search_app import run_episode

    collected: list[EpisodeMetrics] = []
    for scene_id in scene_ids:
        for policy_name in policy_names:
            for seed in seeds:
                started = time.perf_counter()
                trace = run_episode(
                    scene_id=scene_id,
                    scene=scene,
                    seed=seed,
                    policy_name=policy_name,
                    world=world,
                    close_world=world is None,
                )
                collected.append(
                    EpisodeMetrics.from_trace(trace, time.perf_counter() - started)
                )
    return collected


def render_table(episodes: list[EpisodeMetrics]) -> str:
    """Group by (scene, policy) and print one row each."""
    groups: dict[tuple[str, str], list[EpisodeMetrics]] = {}
    for episode in episodes:
        groups.setdefault((episode.scene_id, episode.policy), []).append(episode)

    rows = [AggregateMetrics.over(group) for group in groups.values()]
    header = (
        f"{'scene':<22}{'policy':<15}{'n':>4}{'success':>9}{'fail':>6}{'opened':>8}"
        f"{'repeat':>8}{'unnec':>7}{'travel':>9}{'replan':>8}{'ms':>8}"
    )
    out = [header, "-" * len(header)]
    for row in rows:
        out.append(
            f"{row.scene_id:<22}{row.policy:<15}{row.episodes:>4}"
            f"{row.success_rate:>8.0%}{row.failure_rate:>6.0%}"
            f"{row.mean_containers_opened:>8.2f}"
            f"{row.mean_repeated:>8.2f}{row.mean_unnecessary:>7.2f}"
            f"{row.mean_travel:>9.2f}{row.mean_replans:>8.2f}"
            f"{row.mean_elapsed_ms:>8.1f}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .search_policy import POLICIES
    from .search_scenes import SCENES

    parser = argparse.ArgumentParser(description="D1 metrics sweep over the mock scenes")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--scene", action="append", default=None, choices=sorted(SCENES))
    args = parser.parse_args(argv)

    scene_ids = args.scene or sorted(SCENES)
    episodes = sweep(
        scene_ids=scene_ids,
        policy_names=sorted(POLICIES),
        seeds=range(args.seeds),
    )
    print(render_table(episodes))

    print("\ndeterminism (same seed replays an identical trace):")
    ok = True
    for scene_id in scene_ids:
        for policy_name in sorted(POLICIES):
            good = replay_is_deterministic(
                scene_id=scene_id, seed=11, policy_name=policy_name
            )
            ok = ok and good
            print(f"  {scene_id:<22}{policy_name:<15}{'pass' if good else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
