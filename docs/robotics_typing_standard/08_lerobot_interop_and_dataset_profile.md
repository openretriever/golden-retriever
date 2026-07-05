# Dataset and LeRobot Interop

This page defines Golden's dataset/export profile: event rows, manifests, and an adapter-based bridge to LeRobot-style records. It is for interchange at dataset boundaries, not for changing how core Retriever runs a graph.

## Event Table Shape

`retriever_typing.data.dataset_manifest.EVENT_TABLE_COLUMNS` defines the canonical columns:

| Column | Meaning |
| --- | --- |
| `episode_id` | Episode identity. |
| `stream_id` | Stream identity. |
| `event_time_ns` | Event-time timestamp. |
| `ingest_time_ns` | Ingestion timestamp. |
| `seq` | Per-stream sequence number. |
| `type_name` | Payload type identity. |
| `payload` | Serialized payload. |
| `lineage` | Source event references. |
| `frame_id` | Frame metadata when relevant. |
| `units` | Unit metadata when relevant. |

## Manifest Contracts

| Contract | Role |
| --- | --- |
| `FieldSpec` | One schema field. |
| `StreamSpec` | One typed stream. |
| `DataSpec` | Dataset-level stream/type profile. |
| `EpisodeManifest` | Bounds, stream set, event count, and artifacts for one episode. |
| `DatasetManifest` | Immutable list of episode manifests. |

Helpers:

```python
from retriever_typing.data import (
    build_episode_manifest,
    build_dataset_manifest,
    validate_dataset_manifest,
)
```

## LeRobot Bridge

```python
from retriever_typing.data import to_lerobot_records, from_lerobot_records
```

Mapping principles:

- preserve deterministic order by episode, stream, and frame index,
- keep Retriever metadata under each record's `metadata`,
- preserve lineage, frame, and unit fields for round-trip checks,
- keep the bridge optional so Golden does not require a mandatory LeRobot dependency.

## Run The Demo

```bash
python examples/advanced/robotics_typing_standard/lerobot_bridge_demo.py
```

Expected result: event rows become LeRobot-style records and round-trip back through the bridge with stable metadata.

## Recommended Use

1. Build event rows from `EventBuffer` values.
2. Validate the episode and dataset manifests.
3. Convert to LeRobot records only at interchange boundaries.
4. Keep runtime transport unchanged; use the adapter layer for export/import.
