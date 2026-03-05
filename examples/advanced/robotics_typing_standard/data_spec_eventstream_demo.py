"""Data spec event stream demo.

Shows deterministic event ordering and processing-time style sampling.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from retriever_typing.data import (
    Event,
    EventBuffer,
    StreamId,
    WindowPolicy,
    latest,
    merge_sorted,
    window_agg,
)


def _event(stream: str, t: int, ingest: int, seq: int, value: float) -> Event[float]:
    return Event(
        stream_id=StreamId(stream),
        event_time_ns=t,
        ingest_time_ns=ingest,
        seq=seq,
        value=value,
        type_name="float",
    )


def main() -> None:
    camera = EventBuffer(
        (
            _event("camera", 2_000, 2_100, 1, 0.5),
            _event("camera", 1_000, 1_050, 0, 0.2),
        )
    )
    joint = EventBuffer(
        (
            _event("joint", 1_500, 1_510, 0, 1.0),
            _event("joint", 2_000, 2_020, 1, 1.4),
        )
    )

    merged = merge_sorted(camera, joint)

    print("Deterministic merged order:")
    for event in merged:
        print(
            f"  stream={event.stream_id} event={event.event_time_ns} "
            f"ingest={event.ingest_time_ns} seq={event.seq} value={event.value}"
        )

    print("\nProcessing-time profile:")
    print(f"  latest = {latest(merged)}")
    mean_policy = WindowPolicy(duration_ns=1_000, agg="mean")
    print(f"  window mean at now=2_000ns over 1_000ns = {window_agg(merged, now_ns=2_000, policy=mean_policy)}")


if __name__ == "__main__":
    main()
