"""Belief over which container holds the target.

A discrete distribution across the scene's containers, uniform at reset. Under
noiseless observation the update is exact Bayes and collapses to arithmetic:
looking inside a container and not finding the target drives its mass to zero
and renormalizes the rest; finding it ends the episode.

Deliberately simple. Observation noise — a look that misses a target that is
actually there — belongs to D3's memory benchmark, not to D1.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from retriever.flow import Flow, io

__all__ = ["InspectionResult", "SearchBelief", "BeliefTracker"]

_TOL = 1e-9


@io
@dataclass(frozen=True)
class InspectionResult:
    """What looking inside one container revealed."""

    container_id: str = ""
    found: bool = False
    step: int = 0


@io
@dataclass(frozen=True)
class SearchBelief:
    """Where the target probably is, and where we have already looked."""

    container_ids: tuple[str, ...] = ()
    probabilities: tuple[float, ...] = ()
    inspected: tuple[str, ...] = ()
    found_in: str = ""

    @classmethod
    def uniform(cls, container_ids: tuple[str, ...]) -> SearchBelief:
        if not container_ids:
            raise ValueError("a belief needs at least one container")
        share = 1.0 / len(container_ids)
        return cls(
            container_ids=tuple(container_ids),
            probabilities=tuple(share for _ in container_ids),
        )

    # --- queries ----------------------------------------------------------

    def probability(self, container_id: str) -> float:
        try:
            return self.probabilities[self.container_ids.index(container_id)]
        except ValueError:
            raise KeyError(f"{container_id!r} is not in this belief") from None

    @property
    def found(self) -> bool:
        return bool(self.found_in)

    @property
    def candidates(self) -> tuple[str, ...]:
        """Containers that could still hold the target (non-zero mass)."""
        return tuple(
            cid for cid, p in zip(self.container_ids, self.probabilities) if p > _TOL
        )

    @property
    def exhausted(self) -> bool:
        """Every container has been ruled out and the target was never found."""
        return not self.found and not self.candidates

    @property
    def is_collapsed(self) -> bool:
        """Only one container can still hold the target — no need to look further."""
        return len(self.candidates) == 1

    def most_likely(self) -> str:
        """Highest-mass container, ties broken by declaration order."""
        if not self.candidates:
            raise ValueError("belief is exhausted; no container can hold the target")
        return max(
            zip(self.container_ids, self.probabilities),
            key=lambda pair: (pair[1], -self.container_ids.index(pair[0])),
        )[0]

    # --- update -----------------------------------------------------------

    def update(self, result: InspectionResult) -> SearchBelief:
        """Fold one observation in. Returns a new belief; never mutates."""
        if result.container_id not in self.container_ids:
            raise KeyError(f"{result.container_id!r} is not in this belief")

        inspected = self.inspected
        if result.container_id not in inspected:
            inspected = inspected + (result.container_id,)

        if result.found:
            certain = tuple(
                1.0 if cid == result.container_id else 0.0 for cid in self.container_ids
            )
            return replace(
                self,
                probabilities=certain,
                inspected=inspected,
                found_in=result.container_id,
            )

        cleared = [
            0.0 if cid == result.container_id else p
            for cid, p in zip(self.container_ids, self.probabilities)
        ]
        total = sum(cleared)
        if total > _TOL:
            cleared = [p / total for p in cleared]
        return replace(self, probabilities=tuple(cleared), inspected=inspected)

    def as_dict(self) -> dict[str, float]:
        """Rounded view for traces and printing."""
        return {
            cid: round(p, 6) for cid, p in zip(self.container_ids, self.probabilities)
        }


class BeliefTracker(Flow[InspectionResult, SearchBelief]):
    """Maintain a `SearchBelief` across one search episode."""

    def __init__(self, *, container_ids: tuple[str, ...]):
        super().__init__()
        self.container_ids = tuple(container_ids)

    def init_config(self) -> dict:
        return {"container_ids": list(self.container_ids)}

    def reset(self) -> None:
        self.belief = SearchBelief.uniform(self.container_ids)

    def step(self, result: InspectionResult) -> SearchBelief:
        if result is None or not result.container_id:
            return self.belief
        self.belief = self.belief.update(result)
        return self.belief
