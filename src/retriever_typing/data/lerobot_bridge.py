"""LeRobot profile bridge for retriever_typing.data.v1 rows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence


def to_lerobot_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map canonical event-table rows into LeRobot-compatible records.

    The output keeps Retriever metadata under `metadata` while preserving
    deterministic frame order per (episode_id, stream_id).
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["episode_id"], row["stream_id"])].append(row)

    records: list[dict[str, Any]] = []
    for (episode_id, stream_id), group in grouped.items():
        ordered = sorted(
            group,
            key=lambda row: (
                row["event_time_ns"],
                row["ingest_time_ns"],
                row["stream_id"],
                row["seq"],
            ),
        )
        for frame_index, row in enumerate(ordered):
            records.append(
                {
                    "episode_id": episode_id,
                    "stream_id": stream_id,
                    "frame_index": frame_index,
                    "timestamp_ns": row["event_time_ns"],
                    "type_name": row["type_name"],
                    "payload": row["payload"],
                    "metadata": {
                        "ingest_time_ns": row["ingest_time_ns"],
                        "seq": row["seq"],
                        "frame_id": row.get("frame_id"),
                        "units": row.get("units"),
                        "lineage": row.get("lineage", []),
                    },
                }
            )

    records.sort(key=lambda rec: (rec["episode_id"], rec["stream_id"], rec["frame_index"]))
    return records


def from_lerobot_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map LeRobot-compatible records back into canonical event-table rows."""
    rows: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata", {})
        timestamp_ns = int(record["timestamp_ns"])
        rows.append(
            {
                "episode_id": record["episode_id"],
                "stream_id": record["stream_id"],
                "event_time_ns": timestamp_ns,
                "ingest_time_ns": int(metadata.get("ingest_time_ns", timestamp_ns)),
                "seq": int(metadata.get("seq", record.get("frame_index", 0))),
                "type_name": record["type_name"],
                "payload": record.get("payload"),
                "lineage": metadata.get("lineage", []),
                "frame_id": metadata.get("frame_id"),
                "units": metadata.get("units"),
            }
        )

    rows.sort(
        key=lambda row: (
            row["event_time_ns"],
            row["ingest_time_ns"],
            row["stream_id"],
            row["seq"],
        )
    )
    return rows


def validate_lerobot_mapping(records: Sequence[dict[str, Any]]) -> None:
    """Validate minimal LeRobot mapping integrity."""
    required = {
        "episode_id",
        "stream_id",
        "frame_index",
        "timestamp_ns",
        "type_name",
        "payload",
        "metadata",
    }

    frame_indices: dict[tuple[str, str], list[int]] = defaultdict(list)

    for idx, record in enumerate(records):
        missing = sorted(required.difference(record.keys()))
        if missing:
            raise ValueError(f"record[{idx}] missing keys: {missing}")

        frame_index = int(record["frame_index"])
        if frame_index < 0:
            raise ValueError(f"record[{idx}] frame_index must be >= 0")

        timestamp_ns = int(record["timestamp_ns"])
        if timestamp_ns < 0:
            raise ValueError(f"record[{idx}] timestamp_ns must be >= 0")

        key = (str(record["episode_id"]), str(record["stream_id"]))
        frame_indices[key].append(frame_index)

    for key, indices in frame_indices.items():
        expected = list(range(len(indices)))
        if sorted(indices) != expected:
            raise ValueError(
                f"non-contiguous frame_index for {key}: got {sorted(indices)}, expected {expected}"
            )


__all__ = [
    "from_lerobot_records",
    "to_lerobot_records",
    "validate_lerobot_mapping",
]
