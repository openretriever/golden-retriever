"""Event buffer operators for data spec v1."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .v1 import Event, EventBuffer, WatermarkPolicy, WindowPolicy


def merge_sorted(*buffers: EventBuffer[Any]) -> EventBuffer[Any]:
    """Merge multiple buffers using deterministic event ordering."""
    merged = []
    for buffer in buffers:
        merged.extend(buffer.events)
    return EventBuffer(tuple(sorted(merged, key=lambda event: event.ordering_key())))


def watermark_prune(buffer: EventBuffer[Any], policy: WatermarkPolicy) -> EventBuffer[Any]:
    """Drop events older than watermark-window according to policy."""
    threshold = policy.watermark_ns - policy.allowed_lateness_ns
    if not policy.drop_late:
        return buffer
    return EventBuffer(tuple(event for event in buffer.events if event.event_time_ns >= threshold))


def latest(buffer: EventBuffer[Any]) -> Any:
    """Processing-time latest sample from a buffer."""
    event = buffer.latest_event()
    if event is None:
        raise IndexError("cannot sample latest from empty buffer")
    return event.value


def hold(buffer: EventBuffer[Any], *, now_ns: int, last_value: Optional[Any] = None) -> Optional[Any]:
    """Processing-time hold sample: latest event at or before now_ns, else last_value."""
    if now_ns < 0:
        raise ValueError("now_ns must be >= 0")

    candidates = [event for event in buffer.events if event.event_time_ns <= now_ns]
    if not candidates:
        return last_value
    return sorted(candidates, key=lambda event: event.ordering_key())[-1].value


def window_values(buffer: EventBuffer[Any], *, now_ns: int, duration_ns: int) -> tuple[Any, ...]:
    """Return values in [now_ns - duration_ns, now_ns]."""
    if duration_ns <= 0:
        raise ValueError("duration_ns must be > 0")
    start = now_ns - duration_ns
    window = buffer.within(start_ns=start, end_ns=now_ns)
    ordered = window.sorted()
    return ordered.values()


def window_agg(
    buffer: EventBuffer[Any],
    *,
    now_ns: int,
    policy: WindowPolicy,
    fallback: Optional[Any] = None,
) -> Any:
    """Aggregate a window using first/last/max/min/mean semantics."""
    values = window_values(buffer, now_ns=now_ns, duration_ns=policy.duration_ns)
    if not values:
        return fallback

    if policy.agg == "first":
        return values[0]
    if policy.agg == "last":
        return values[-1]
    if policy.agg == "max":
        return max(values)
    if policy.agg == "min":
        return min(values)
    if policy.agg == "mean":
        total = 0.0
        for value in values:
            total += float(value)
        return total / len(values)

    raise ValueError(f"unsupported aggregation: {policy.agg}")


def from_events(events: Iterable[Event[Any]]) -> EventBuffer[Any]:
    """Create a deterministic buffer from raw events."""
    return EventBuffer(tuple(events)).sorted()


__all__ = [
    "from_events",
    "hold",
    "latest",
    "merge_sorted",
    "watermark_prune",
    "window_agg",
    "window_values",
]
