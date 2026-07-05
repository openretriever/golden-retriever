# Robot Data Profile + EventStream v1 (`retriever_typing.data`)

<div class="gr-route-pills gr-route-pills-inline">
  <a href="https://openretriever.org/">Retriever home</a>
  <a href="https://openretriever.org/start/">Start path</a>
  <a href="https://openretriever-docs.pages.dev/">Core docs</a>
  <a href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">Visual quickstart</a>
  <a href="https://github.com/openretriever/retriever">Core source</a>
  <a href="/">Golden overview</a>
  <a href="https://github.com/openretriever/golden-retriever">Golden source</a>
  <a href="../llms.txt">Golden agent map</a>
</div>

Golden owns this robot-facing profile; core Retriever owns the runtime, clocks, Flow/Pipeline semantics, standard type registry, and Hub loading mechanics. Use these pages when you need reusable robotics payloads or dataset/event profiles on top of the core runtime.


## Goal

Define a reusable, deterministic data contract for collection/replay/export workflows without changing default core runtime behavior.

Canonical package:
- `retriever_typing.data`
- pinned path: `retriever_typing.data.v1`

## Core contracts

- `Event[T]`
  - immutable event record with deterministic ordering key:
  - `(event_time_ns, ingest_time_ns, stream_id, seq)`
- `EventRef`, `LineageRef`
  - explicit source lineage for derived/joined events.
- `StreamId`, `ClockDomain`, `SchemaRef`
  - stable stream identity and schema metadata.
- `EventBuffer[T]`, `MultiStreamBuffer`
  - immutable stream buffers.
- `JoinPolicy`, `WatermarkPolicy`, `WindowPolicy`
  - explicit policy objects for alignment/pruning/sampling.

## Deterministic multi-stream operators

Event-time profile (normative):
- `align_exact`
- `align_latest_before(max_delta_ns)`
- `align_window(window_ns)`
- `merge_sorted`
- `watermark_prune`

Processing-time compatibility profile:
- `latest`
- `hold`
- `window_agg`

## Minimal runtime impact

This page does not modify core Retriever scheduler/runtime behavior.

Interop is opt-in via:
- `from_runtime_event_buffer(...)`
- `to_runtime_event_buffer(...)`

## Example imports

```python
from retriever_typing.data import Event, EventBuffer, align_latest_before, WindowPolicy
from retriever_typing.data.v1 import EventBuffer as PinnedEventBuffer
```

## Acceptance checks

- import contract works for both convenience and pinned paths,
- mixed-stream merges are deterministic,
- join/window semantics are stable and test-covered,
- no dependency from `src/retriever_typing` back into legacy system packages.
