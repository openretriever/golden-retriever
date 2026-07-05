# Robot Dataset Profile + LeRobot Interop (`retriever_typing.data`)

<div class="gr-route-pills gr-route-pills-inline">
  <a href="https://openretriever.org/">Retriever home</a>
  <a href="https://openretriever-docs.pages.dev/">Core docs</a>
  <a href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">Visual quickstart</a>
  <a href="https://github.com/openretriever/retriever">Core source</a>
  <a href="/">Golden overview</a>
  <a href="https://github.com/openretriever/golden-retriever">Golden source</a>
  <a href="../llms.txt">Golden agent map</a>
</div>

Golden owns this robot-facing profile; core Retriever owns the runtime, clocks, Flow/Pipeline semantics, standard type registry, and Hub loading mechanics. Use these pages when you need reusable robotics payloads or dataset/event profiles on top of the core runtime.


## Goal

Standardize replay/export metadata and provide an adapter-based mapping to LeRobot-style records.

## Canonical event table schema

`retriever_typing.data.dataset_manifest.EVENT_TABLE_COLUMNS`:
- `episode_id`
- `stream_id`
- `event_time_ns`
- `ingest_time_ns`
- `seq`
- `type_name`
- `payload`
- `lineage`
- `frame_id`
- `units`

## Manifest contracts

- `FieldSpec`, `StreamSpec`, `DataSpec`
  - schema-level stream/type contracts.
- `EpisodeManifest`
  - per-episode bounds, stream set, event count, artifacts.
- `DatasetManifest`
  - dataset-level metadata with immutable episode list.

Helpers:
- `build_episode_manifest(...)`
- `build_dataset_manifest(...)`
- `validate_dataset_manifest(...)`

## LeRobot bridge API

- `to_lerobot_records(rows)`
- `from_lerobot_records(records)`
- `validate_lerobot_mapping(records)`

Mapping principles:
- deterministic ordering per `(episode_id, stream_id, frame_index)`,
- retain Retriever metadata under record `metadata`,
- preserve lineage/frame/unit fields for roundtrip.

## Non-goals in this wave

- no forced change to retriever core runtime serde path,
- no mandatory LeRobot schema dependency,
- no requirement that all payloads become robotics-v1 envelopes.

## Recommended usage

1. Build event rows from `EventBuffer`.
2. Validate manifests and row contracts.
3. Convert to LeRobot records only at dataset interchange boundaries.
4. Keep core runtime transport unchanged; use adapter layer for export/import.
