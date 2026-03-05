"""Multi-stream join demo for event-time contracts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from retriever_typing.data import (
    Event,
    EventBuffer,
    StreamId,
    align_exact,
    align_latest_before,
    align_window,
)


def _event(stream: str, t: int, seq: int, value: int) -> Event[int]:
    return Event(
        stream_id=StreamId(stream),
        event_time_ns=t,
        ingest_time_ns=t,
        seq=seq,
        value=value,
        type_name="int",
    )


def _print(label: str, buffer: EventBuffer[tuple[int, int]]) -> None:
    print(label)
    if not buffer:
        print("  <empty>")
        return
    for event in buffer:
        lineage = event.lineage.sources if event.lineage else ()
        print(
            f"  t={event.event_time_ns} value={event.value} "
            f"lineage={[f'{src.stream_id}:{src.event_time_ns}' for src in lineage]}"
        )


def main() -> None:
    left = EventBuffer((_event("A", 100, 0, 1), _event("A", 180, 1, 2), _event("A", 260, 2, 3)))
    right = EventBuffer((_event("B", 180, 0, 10), _event("B", 200, 1, 20), _event("B", 450, 2, 30)))

    _print("Exact join:", align_exact(left, right))
    _print("Latest-before join (max_delta=50):", align_latest_before(left, right, max_delta_ns=50))
    _print("Window join (+/-20):", align_window(left, right, window_ns=20))


if __name__ == "__main__":
    main()
