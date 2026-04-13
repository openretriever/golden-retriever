"""Local memory payloads for the concise advanced memory ladder."""

from __future__ import annotations

from dataclasses import dataclass

from retriever.flow import io


@dataclass(frozen=True)
class ObjectBelief:
    label: str
    x_norm: float
    y_norm: float
    confidence: float
    seen_count: int
    last_frame_index: int
    missing_steps: int


@io
@dataclass(frozen=True)
class SceneBelief:
    frame_index: int | None = None
    objects: tuple[ObjectBelief, ...] = ()
