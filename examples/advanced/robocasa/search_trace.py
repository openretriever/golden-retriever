"""The search trace: an append-only record of what the robot decided and saw.

The D1 done-definition requires the full search to be inspectable "without
reading internal simulator code", so the trace is a standalone artifact: one
JSON object per line, plus a renderer that prints it legibly. Nothing here
imports a simulator, and a trace can be read back and re-rendered by anyone.

Each record carries the belief both before and after the observation, which is
what makes the belief update auditable rather than asserted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["TraceRecord", "SearchTrace"]


@dataclass(frozen=True)
class TraceRecord:
    """One step of the search."""

    step: int
    event: str  # episode_started | planned | inspected | terminated | failed
    container_id: str = ""
    plan: dict = field(default_factory=dict)
    found: bool = False
    reason: str = ""
    belief_before: dict[str, float] = field(default_factory=dict)
    belief_after: dict[str, float] = field(default_factory=dict)
    repeated: bool = False
    travel_cost: float = 0.0  # distance for this leg, from the previous stop
    progress: float = 0.0  # fraction of the container space resolved, 0.0-1.0
    status: str = ""  # IN_PROGRESS | SUCCESS | FAILURE
    error_message: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class SearchTrace:
    """Everything one episode did, in order."""

    scene_id: str
    seed: int
    policy: str
    target: str
    records: list[TraceRecord] = field(default_factory=list)

    def append(self, record: TraceRecord) -> TraceRecord:
        self.records.append(record)
        return record

    # --- derived facts ----------------------------------------------------

    @property
    def inspections(self) -> list[TraceRecord]:
        return [r for r in self.records if r.event == "inspected"]

    @property
    def containers_opened(self) -> int:
        """Distinct containers looked inside."""
        return len({r.container_id for r in self.inspections})

    @property
    def repeated_inspections(self) -> int:
        """Looks into a container already inspected this episode."""
        return sum(1 for r in self.inspections if r.repeated)

    @property
    def total_travel_cost(self) -> float:
        """Distance walked across the whole search.

        Unlike `containers_opened`, this one genuinely separates policies: with
        a uniformly placed target, visit *order* cannot change how many
        containers you open in expectation, but it very much changes how far
        you walk to open them.
        """
        return sum(r.travel_cost for r in self.inspections)

    @property
    def succeeded(self) -> bool:
        return any(r.found for r in self.records)

    @property
    def status(self) -> str:
        """The episode's terminal status: SUCCESS, FAILURE, or IN_PROGRESS."""
        for record in reversed(self.records):
            if record.status:
                return record.status
        return "IN_PROGRESS"

    @property
    def failed(self) -> bool:
        return self.status == "FAILURE"

    @property
    def error_message(self) -> str:
        """Why it failed, empty when it did not."""
        for record in reversed(self.records):
            if record.error_message:
                return record.error_message
        return ""

    @property
    def final_progress(self) -> float:
        return self.records[-1].progress if self.records else 0.0

    @property
    def plans(self) -> list[TraceRecord]:
        return [r for r in self.records if r.event == "planned"]

    @property
    def initial_plan(self) -> list[str]:
        return list(self.plans[0].plan.get("container_ids", [])) if self.plans else []

    @property
    def replans(self) -> int:
        """Times the remaining order stopped matching the prediction.

        The catalog asks for at least one system metric alongside task success
        and names replans first. Zero means the robot followed its opening plan
        exactly; higher means new information kept changing its mind.
        """
        return max(0, len(self.plans) - 1)

    @property
    def followed_initial_plan(self) -> bool:
        """Did the route actually taken match the opening plan's prefix?"""
        opened = [r.container_id for r in self.inspections]
        return opened == self.initial_plan[: len(opened)]

    @property
    def steps(self) -> int:
        return len(self.inspections)

    # --- serialization ----------------------------------------------------

    def header(self) -> dict:
        return {
            "record": "header",
            "scene_id": self.scene_id,
            "seed": self.seed,
            "policy": self.policy,
            "target": self.target,
        }

    def to_jsonl(self) -> str:
        lines = [json.dumps(self.header(), sort_keys=True)]
        lines.extend(r.to_json() for r in self.records)
        return "\n".join(lines) + "\n"

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_jsonl(), encoding="utf-8")
        return destination

    # --- human-readable ---------------------------------------------------

    def render(self) -> str:
        """Print the episode so a reader can follow the reasoning."""
        out = [
            f"scene={self.scene_id} seed={self.seed} policy={self.policy} "
            f"target={self.target!r}",
            "",
        ]
        for record in self.records:
            if record.event == "episode_started":
                belief = _format_belief(record.belief_after)
                out.append(f"  start           belief {belief}")
            elif record.event == "planned":
                order = " -> ".join(record.plan.get("container_ids", [])) or "(nothing)"
                label = "plan" if record.plan.get("revision", 0) == 0 else "replan"
                out.append(f"  {label:<15} {order}")
            elif record.event == "inspected":
                verdict = "FOUND" if record.found else "empty"
                flag = " (repeat!)" if record.repeated else ""
                out.append(
                    f"  step {record.step:<2} open {record.container_id:<16} -> {verdict}{flag}"
                    f"  (+{record.travel_cost:.2f}m, {record.progress:.0%})"
                )
                out.append(f"          why: {record.reason}")
                out.append(f"          belief {_format_belief(record.belief_after)}")
            elif record.event == "terminated":
                out.append(f"  stop            {record.reason}  [{record.status}]")
            elif record.event == "failed":
                out.append(f"  FAILED          {record.error_message}")
        out.extend(
            [
                "",
                f"  status={self.status} success={self.succeeded} "
                f"containers_opened={self.containers_opened} "
                f"repeated={self.repeated_inspections} steps={self.steps} "
                f"travel={self.total_travel_cost:.2f}m progress={self.final_progress:.0%} "
                f"replans={self.replans}",
            ]
        )
        return "\n".join(out)


def _format_belief(belief: dict[str, float]) -> str:
    if not belief:
        return "{}"
    return " ".join(f"{cid}={p:.2f}" for cid, p in belief.items())
