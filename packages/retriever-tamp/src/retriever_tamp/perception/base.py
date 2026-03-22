from __future__ import annotations

from typing import Any, Protocol

from retriever_tamp.core.types import WorldSnapshot


class ObservationReceiver(Protocol):
    def receive(self) -> Any:
        """Return a raw observation from a simulator, robot, or runtime stream."""


class StateExtractor(Protocol):
    def extract(self, observation: Any) -> WorldSnapshot:
        """Convert an observation into a task-relevant snapshot."""


class BeliefUpdater(Protocol):
    def update(self, previous: WorldSnapshot | None, observation: Any) -> WorldSnapshot:
        """Update a belief/snapshot when history matters."""
