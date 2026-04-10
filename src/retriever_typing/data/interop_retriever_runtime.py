"""Opt-in interop adapters for retriever runtime EventBuffer structures.

This module intentionally avoids mutating retriever runtime behavior. It only
converts between runtime tuple buffers and data-spec event buffers.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

from .v1 import Event, EventBuffer, StreamId


def from_runtime_event_buffer(
    runtime_buffer: Sequence[tuple[float, Any]],
    *,
    stream_id: str,
    frame_id: Optional[str] = None,
    units: Optional[str] = None,
    ingest_offset_ns: int = 0,
) -> EventBuffer[Any]:
    """Convert retriever runtime `(timestamp_sec, value)` tuples to EventBuffer."""
    if ingest_offset_ns < 0:
        raise ValueError("ingest_offset_ns must be >= 0")

    events = []
    for seq, (timestamp_sec, value) in enumerate(runtime_buffer):
        event_time_ns = int(float(timestamp_sec) * 1_000_000_000)
        ingest_time_ns = event_time_ns + ingest_offset_ns
        events.append(
            Event(
                stream_id=StreamId.parse(stream_id),
                event_time_ns=event_time_ns,
                ingest_time_ns=ingest_time_ns,
                seq=seq,
                value=value,
                type_name=type(value).__name__,
                frame_id=frame_id,
                units=units,
            )
        )
    return EventBuffer(tuple(events)).sorted()


def to_runtime_event_buffer(buffer: EventBuffer[Any]) -> list[tuple[float, Any]]:
    """Convert data-spec EventBuffer back to runtime `(timestamp_sec, value)` tuples."""
    return [
        (event.event_time_ns / 1_000_000_000.0, event.value)
        for event in buffer.sorted()
    ]


def is_runtime_event_buffer(value: Any) -> bool:
    """Best-effort shape check for runtime EventBuffer-like data."""
    if not isinstance(value, (list, tuple)):
        return False
    if not value:
        return True
    first = value[0]
    return isinstance(first, (tuple, list)) and len(first) == 2


__all__ = [
    "from_runtime_event_buffer",
    "is_runtime_event_buffer",
    "to_runtime_event_buffer",
]
