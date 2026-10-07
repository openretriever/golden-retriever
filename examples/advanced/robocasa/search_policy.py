"""Search policies: given a belief, decide where to look next — or stop.

Every policy answers the same question at container granularity ("inspect
`drawer_middle`" / "stop"), never at joint or action-chunk granularity. Turning
a chosen container into motion is the skill executor's job, which keeps the
search logic simulator-free and testable.

Two policies ship so the metrics mean something. A single number in isolation
says nothing; `containers opened` is only interesting next to a baseline that
opens more of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from retriever.flow import io

from .search_belief import SearchBelief
from .search_scenes import SearchScene

__all__ = [
    "SearchDecision",
    "SearchPolicy",
    "FixedOrderPolicy",
    "NearestFirstPolicy",
    "POLICIES",
    "get_policy",
]

STOP = "stop"
INSPECT = "inspect"


@io
@dataclass(frozen=True)
class SearchDecision:
    """One decision, with the reason recorded so a trace explains itself."""

    action: str = INSPECT
    container_id: str = ""
    reason: str = ""

    @property
    def stops(self) -> bool:
        return self.action == STOP


@runtime_checkable
class SearchPolicy(Protocol):
    """Belief in, next container out.

    `position` is where the robot is now; policies that ignore geometry ignore
    it. It defaults to the scene's robot home so a caller that does not track
    position still gets sensible behaviour.
    """

    name: str

    def decide(
        self,
        belief: SearchBelief,
        scene: SearchScene,
        position: tuple[float, float, float] | None = None,
    ) -> SearchDecision: ...


def _terminal_decision(belief: SearchBelief) -> SearchDecision | None:
    """Stop conditions every policy shares."""
    if belief.found:
        return SearchDecision(STOP, belief.found_in, "target found")
    if belief.exhausted:
        return SearchDecision(STOP, "", "every container ruled out")
    return None


class FixedOrderPolicy:
    """Open containers in the order the scene declares them.

    The baseline. It ignores everything the belief knows except which
    containers are already ruled out.
    """

    name = "fixed-order"

    def decide(
        self,
        belief: SearchBelief,
        scene: SearchScene,
        position: tuple[float, float, float] | None = None,
    ) -> SearchDecision:
        terminal = _terminal_decision(belief)
        if terminal is not None:
            return terminal
        for container_id in scene.container_ids:
            if container_id in belief.candidates:
                return SearchDecision(
                    INSPECT, container_id, "first un-ruled-out container in scene order"
                )
        return SearchDecision(STOP, "", "no candidate remains")


class NearestFirstPolicy:
    """Open the cheapest candidate from wherever the robot currently stands.

    Same belief and same stopping rule as the baseline, so it opens the same
    number of containers in expectation — with a uniformly placed target,
    ordering cannot change that. What it changes is distance travelled, which
    is the metric this policy is actually for.
    """

    name = "nearest-first"

    def decide(
        self,
        belief: SearchBelief,
        scene: SearchScene,
        position: tuple[float, float, float] | None = None,
    ) -> SearchDecision:
        terminal = _terminal_decision(belief)
        if terminal is not None:
            return terminal
        candidates = [c for c in scene.container_ids if c in belief.candidates]
        if not candidates:
            return SearchDecision(STOP, "", "no candidate remains")
        origin = position if position is not None else scene.robot_home
        # Ties break by scene order, so the choice stays deterministic.
        chosen = min(
            candidates,
            key=lambda cid: (scene.travel_cost(cid, origin), scene.container_ids.index(cid)),
        )
        return SearchDecision(
            INSPECT,
            chosen,
            f"nearest candidate, cost={scene.travel_cost(chosen, origin):.2f}",
        )


POLICIES: dict[str, type] = {
    FixedOrderPolicy.name: FixedOrderPolicy,
    NearestFirstPolicy.name: NearestFirstPolicy,
}


def get_policy(name: str) -> SearchPolicy:
    try:
        return POLICIES[name]()
    except KeyError:
        known = ", ".join(sorted(POLICIES))
        raise KeyError(f"unknown policy {name!r}; known policies: {known}") from None
