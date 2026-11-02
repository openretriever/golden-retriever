"""Sweep task x episode x method, and report what the study reports.

One JSON object to stdout and a meaningful exit code, following this folder's
other entry points. The sweep layers on `MethodHarness` rather than replacing
it: the harness already dispatches one control step per chunk row and emits a
`chunk_dispatched` event per `predict`, which is where the two step counts come
from.

Three counts are kept apart because they are different quantities, and the study
reports the third. **Control steps** are rows actually dispatched to the
simulator. **Model calls** are `predict` invocations, one per chunk — the metric
the study names, and the one a longer horizon reduces. **Episodes** are trials.
Nothing here sums them.

Latency is measured but kept out of any canonical record, for the reason the
sibling drawer harness documents: a measured duration differs on every run, so a
record containing one can never be compared byte for byte. It is reported per
episode as a summary and belongs in a sidecar if it is ever joined to evidence.

The protocol follows the study's: tasks x 5 episodes at fixed seeds 0-4 by
default, since that is what the published numbers use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter

#: The study's episode protocol: five episodes per task at fixed seeds.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class MethodSpec:
    """A named method plus how to build a fresh one for each trial.

    A factory rather than an instance: a method may carry per-episode state —
    a scripted phase, a random generator, a call ledger — and reusing one across
    trials would leak an episode into the next.
    """

    method_id: str
    build: object
    representation: str = "eef_delta"
    horizon: int = 1
    label: str = ""

    def __post_init__(self) -> None:
        if not self.method_id or self.method_id != self.method_id.strip():
            raise ValueError("method_id must be a non-empty, unpadded string")
        if not callable(self.build):
            raise TypeError("build must be callable and return a fresh method")


@dataclass
class EpisodeResult:
    """One trial's measurements, in the study's vocabulary."""

    task: str
    method_id: str
    seed: int
    success: bool
    status: str
    score: int
    rungs: dict
    control_steps: int
    model_calls: int
    horizon: int
    representation: str
    latency_s: float
    clip: dict
    backend: dict
    #: What the method reports about itself — for a model arm, the refusal and
    #: retry counts. Without it a zero score cannot be told apart from a run of
    #: held no-motion actions, which are different findings.
    method: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class Sweep:
    """Run every (task, seed, method) cell and summarise per method."""

    build_environment: object
    tasks: tuple = ("PickPlaceCounterToCabinet",)
    seeds: tuple = DEFAULT_SEEDS
    max_steps: int = 400
    #: Per-task episode budgets, overriding `max_steps`. The official protocol
    #: gives each task its own horizon (450 to 1050 across atomic-seen), so a
    #: flat cap either starves the long tasks or flatters the short ones.
    horizons: dict = field(default_factory=dict)
    #: Action sources the envelope admits. Scripted only by default, so
    #: admitting a model's output is a deliberate, recorded widening rather
    #: than a method quietly mislabelling itself to get through.
    allowed_sources: tuple = ("scripted",)
    #: Append each finished episode to this JSONL and reuse what is already in
    #: it. A full sweep is hours of simulation and the summary is printed only
    #: at the end, so without this an interruption at 90% loses everything.
    #: Keyed on (task, seed, method_id), so a resumed run re-executes exactly
    #: the cells that did not finish.
    checkpoint: str = ""
    results: list = field(default_factory=list)
    #: How many cells came from the checkpoint rather than being run.
    resumed: int = 0

    def run(self, methods) -> dict:
        from .method_harness import MethodHarness, SafetyEnvelope, TrialRequest

        specs = list(methods)
        if not specs:
            raise ValueError("a sweep needs at least one method")
        if len({spec.method_id for spec in specs}) != len(specs):
            raise ValueError("methods contain a duplicate method_id")
        harness = MethodHarness(safety=SafetyEnvelope(
            allowed_sources=frozenset(self.allowed_sources)))
        finished = self._finished()
        for task in self.tasks:
            for seed in self.seeds:
                for spec in specs:
                    key = (task, int(seed), spec.method_id)
                    if key in finished:
                        self.results.append(finished[key])
                        self.resumed += 1
                        continue
                    result = self._cell(harness, task, seed, spec, TrialRequest)
                    self.results.append(result)
                    self._record(result)
        return self.summarise()

    def _finished(self) -> dict:
        """Episodes already in the checkpoint, keyed by cell."""
        import json
        from pathlib import Path

        if not self.checkpoint:
            return {}
        path = Path(self.checkpoint)
        if not path.exists():
            return {}
        done = {}
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                result = EpisodeResult(**row)
            except (ValueError, TypeError) as error:
                # A truncated last line is the normal cost of being killed
                # mid-write; anything else is a checkpoint worth not trusting.
                raise ValueError(
                    f"{path}:{number} is not a readable episode record: {error}") from error
            done[(result.task, int(result.seed), result.method_id)] = result
        return done

    def _record(self, result) -> None:
        """Append one finished episode, flushed so a kill cannot lose it."""
        import json
        from pathlib import Path

        if not self.checkpoint:
            return
        path = Path(self.checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
            handle.flush()

    def _cell(self, harness, task, seed, spec, TrialRequest):
        environment = self.build_environment(task=task, representation=spec.representation)
        method = spec.build(environment)
        # Tie the method's own randomness to the episode. The environment already
        # receives `seed` and so lays out an identical scene, but a sampling
        # policy draws fresh noise per call, so without this the policy half of
        # the episode is unreproducible while the simulator half is not. Only
        # methods that declare a seeded reset are touched.
        reset = getattr(method, "reset", None)
        if callable(reset):
            try:
                reset(seed)
            except TypeError:
                pass
        started = perf_counter()
        report = harness.run(
            TrialRequest(method_id=spec.method_id, task=task, episode=seed, seed=seed,
                         max_steps=int(self.horizons.get(task, self.max_steps))),
            environment=environment, method=method, close_environment=False)
        latency = perf_counter() - started
        # One `chunk_dispatched` event per `predict`, which is the study's
        # model-call count and is read from the harness rather than from the
        # method, so a method cannot misreport it.
        calls = sum(1 for event in report.events if event.kind == "chunk_dispatched")
        progress = getattr(environment, "progress", None)
        codec = getattr(environment, "codec", None)
        described = {}
        describe = getattr(method, "describe", None)
        if callable(describe):
            try:
                value = describe()
            except Exception:  # noqa: BLE001 - a method's own reporting must not lose a cell
                value = None
            if isinstance(value, dict):
                described = {k: v for k, v in value.items()
                             if isinstance(v, (int, float, str, bool, type(None)))}
        for name in ("refusals", "last_refusal"):
            if hasattr(method, name):
                described.setdefault(name, getattr(method, name))
        client = getattr(method, "client", None)
        client_describe = getattr(client, "describe", None)
        if callable(client_describe):
            try:
                value = client_describe()
            except Exception:  # noqa: BLE001
                value = None
            if isinstance(value, dict):
                described.update({f"client_{k}": v for k, v in value.items()
                                  if isinstance(v, (int, float, str, bool, type(None)))})
        # Captured before close(): a closed environment reports nothing.
        backend_described = environment.describe()
        try:
            environment.close()
        except Exception:  # noqa: BLE001 - cleanup failure must not lose the result
            pass
        return EpisodeResult(
            task=task, method_id=spec.method_id, seed=seed,
            success=bool(report.success), status=report.status,
            score=int(progress.score) if progress else 0,
            rungs=dict(progress.rungs) if progress else {},
            control_steps=int(report.steps), model_calls=calls, horizon=spec.horizon,
            representation=spec.representation, latency_s=round(latency, 4),
            clip=codec.ledger.to_dict() if codec else {},
            backend=backend_described, method=described, error=report.error)

    def summarise(self) -> dict:
        per_method: dict = {}
        for result in self.results:
            bucket = per_method.setdefault(result.method_id, [])
            bucket.append(result)
        summary = {}
        for method_id, rows in sorted(per_method.items()):
            count = len(rows)
            summary[method_id] = {
                "episodes": count,
                "successes": sum(row.success for row in rows),
                # The study's two headline numbers.
                "success_rate": round(sum(row.success for row in rows) / count, 4),
                "mean_score": round(sum(row.score for row in rows) / count, 2),
                # Three counts, reported separately and never summed.
                "control_steps": sum(row.control_steps for row in rows),
                "model_calls": sum(row.model_calls for row in rows),
                "mean_control_steps": round(sum(row.control_steps for row in rows) / count, 2),
                "mean_model_calls": round(sum(row.model_calls for row in rows) / count, 2),
                "mean_latency_s": round(sum(row.latency_s for row in rows) / count, 3),
                "rungs_reached": {
                    rung: sum(bool(row.rungs.get(rung)) for row in rows)
                    for rung in (rows[0].rungs or {})},
                "clipped_fraction": _mean_clip(rows),
                "errors": sum(1 for row in rows if row.error),
            }
        return {"schema_version": 1,
                "protocol": {"tasks": list(self.tasks), "seeds": list(self.seeds),
                             "max_steps": self.max_steps,
                             "horizons": dict(sorted(self.horizons.items())),
                             "allowed_sources": sorted(self.allowed_sources),
                             "resumed_episodes": self.resumed,
                             "episodes": len(self.tasks) * len(self.seeds)},
                "episodes": [row.to_dict() for row in self.results],
                "summary": summary,
                # A sweep of baselines is expected to score zero, so "passed"
                # would be a meaningless verdict. Errors are the thing a run
                # can fail on.
                "errors": sum(1 for row in self.results if row.error)}


def _mean_clip(rows):
    fractions = [row.clip.get("clipped_fraction") for row in rows
                 if row.clip.get("clipped_fraction") is not None]
    return round(sum(fractions) / len(fractions), 4) if fractions else None


def render(summary: dict, *, summary_only: bool = False) -> str:
    payload = ({key: value for key, value in summary.items() if key != "episodes"}
               if summary_only else summary)
    return json.dumps(payload, sort_keys=True, allow_nan=False)
