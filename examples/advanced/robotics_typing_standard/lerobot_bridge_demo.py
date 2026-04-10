"""LeRobot bridge demo for retriever_typing.data rows."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from retriever_typing.data import (
    Event,
    EventBuffer,
    StreamId,
    event_table_rows,
    from_lerobot_records,
    to_lerobot_records,
    validate_lerobot_mapping,
)


def _event(stream: str, t: int, seq: int, payload: dict[str, float]) -> Event[dict[str, float]]:
    return Event(
        stream_id=StreamId(stream),
        event_time_ns=t,
        ingest_time_ns=t + 10,
        seq=seq,
        value=payload,
        type_name="dict",
    )


def main() -> None:
    buffer = EventBuffer(
        (
            _event("joint", 1_000, 0, {"q1": 0.1}),
            _event("joint", 2_000, 1, {"q1": 0.2}),
            _event("joint", 3_000, 2, {"q1": 0.3}),
        )
    )

    rows = event_table_rows(buffer, episode_id="episode-001")
    records = to_lerobot_records(rows)
    validate_lerobot_mapping(records)
    roundtrip_rows = from_lerobot_records(records)

    print(f"Canonical rows: {len(rows)}")
    print(f"LeRobot records: {len(records)}")
    print(f"Roundtrip rows: {len(roundtrip_rows)}")
    print("Sample record:")
    print(f"  {records[0]}")


if __name__ == "__main__":
    main()
