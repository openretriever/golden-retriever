"""Window utilities for event-time and processing-time profiles."""

from __future__ import annotations

from typing import Any, Optional

from .buffer import window_agg
from .v1 import EventBuffer, WindowPolicy


def event_window(
    buffer: EventBuffer[Any],
    *,
    now_ns: int,
    duration_ns: int,
) -> EventBuffer[Any]:
    """Return event-time slice [now_ns - duration_ns, now_ns]."""
    if duration_ns <= 0:
        raise ValueError("duration_ns must be > 0")
    start_ns = now_ns - duration_ns
    return buffer.within(start_ns=start_ns, end_ns=now_ns).sorted()


def processing_window_agg(
    buffer: EventBuffer[Any],
    *,
    now_ns: int,
    policy: WindowPolicy,
    fallback: Optional[Any] = None,
) -> Any:
    """Apply processing-time compatible window aggregation semantics."""
    return window_agg(buffer, now_ns=now_ns, policy=policy, fallback=fallback)


__all__ = ["event_window", "processing_window_agg"]
