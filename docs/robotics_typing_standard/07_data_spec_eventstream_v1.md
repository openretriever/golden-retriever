# Data and Event Streams

Golden's data profile is for collection, replay, joins, and dataset export. It is not a replacement for the core Retriever scheduler. It gives applied examples a deterministic event-time record format when a run needs to become evidence.

## Core Contracts

```python
from retriever_typing.data import Event, EventBuffer, align_latest_before
```

| Contract | Role |
| --- | --- |
| `Event[T]` | Immutable value with `event_time_ns`, `ingest_time_ns`, `stream_id`, `seq`, payload, schema, frame, units, and lineage. |
| `EventBuffer[T]` | Immutable ordered collection with `sorted()`, `latest_value()`, and window helpers. |
| `MultiStreamBuffer` | Named collection of stream buffers. |
| `EventRef` / `LineageRef` | Explicit source references for derived events. |
| `StreamId`, `ClockDomain`, `SchemaRef` | Stable stream identity and schema metadata. |

## Deterministic Operators

| Operator | Use |
| --- | --- |
| `merge_sorted` | Merge streams by `(event_time_ns, ingest_time_ns, stream_id, seq)`. |
| `align_exact` | Join events only at matching event times. |
| `align_latest_before(max_delta_ns)` | Join the most recent upstream event before the target time. |
| `align_window(window_ns)` | Gather a bounded event-time window for aggregation. |
| `watermark_prune` | Drop events older than a deterministic watermark. |

## Runtime Boundary

The data profile is opt-in. It does not change default Retriever runtime scheduling.

Use these helpers only at conversion boundaries:

```python
from retriever_typing.data import from_runtime_event_buffer, to_runtime_event_buffer
```

## Run The Demos

```bash
python examples/advanced/robotics_typing_standard/data_spec_eventstream_demo.py
python examples/advanced/robotics_typing_standard/multi_stream_join_demo.py
```

Expected result: deterministic ordering, join/window output, and lineage that points back to source events.

## When To Use This

Use the data profile when you need replayable evidence, dataset export, or stable multi-stream joins. For a small one-off Flow that only consumes its latest input, keep the payload simple and stay with the core runtime APIs.
