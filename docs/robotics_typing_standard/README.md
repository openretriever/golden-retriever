# Robot Type Packs

<div class="gr-compact-hero">
  <p class="gr-eyebrow">Applied payload layer</p>
  <h1>Reusable robot types for Golden examples.</h1>
  <p>Core Retriever teaches the runtime. Golden keeps the robot-facing payload vocabulary that examples can share: world state, belief, skills, plans, stamped spatial values, event buffers, and dataset/export profiles.</p>
</div>

<div class="gr-action-grid gr-action-grid-wide">
  <a class="gr-action-card" href="01_type_catalog_and_semantics/">
    <span>Catalog</span>
    <strong>Know the payloads</strong>
    <small>Spatial values, stamped wrappers, plans, trajectories, status, and validation boundaries.</small>
  </a>
  <a class="gr-action-card" href="02_flow_composition_contract/">
    <span>Composition</span>
    <strong>Wire them safely</strong>
    <small>Composite Flow I/O, qualified field access, and collision behavior for reusable robot graphs.</small>
  </a>
  <a class="gr-action-card" href="07_data_spec_eventstream_v1/">
    <span>Data</span>
    <strong>Record and replay streams</strong>
    <small>Event buffers, event-time joins, windows, lineage, and deterministic dataset export.</small>
  </a>
</div>

## Start Here

Run the three public checks first. They are short and do not require a robot, camera, simulator, or network service.

```bash
pixi run demo-robotics-typing-catalog
pixi run demo-robotics-typing-contract
pixi run demo-robotics-typing-boundary
```

Expected result: terminal output showing stamped spatial payloads, composite I/O access rules, and a perception-to-control boundary that preserves frame, time, and source metadata.

## What This Pack Owns

| Area | Golden owns | Core Retriever owns |
| --- | --- | --- |
| Spatial payload use | Robot-facing examples and validation walkthroughs | Canonical spatial type definitions in the runtime |
| Robot task payloads | `WorldState`, `BeliefGraph`, `Skill`, `Plan`, `Trajectory`, `ExecutionStatus` | Flow/Pipeline execution, clocks, sync, IR, replay |
| Event/data profile | `Event`, `EventBuffer`, manifests, LeRobot bridge examples | Runtime event-buffer mechanics and scheduler behavior |
| Hub readiness | Pack boundary, export catalog, smoke demos | Hub loader and registry mechanics |

## Read Next

1. [Type catalog](01_type_catalog_and_semantics.md) for the actual payload vocabulary.
2. [Flow composition contract](02_flow_composition_contract.md) for reusable graph wiring rules.
3. [Data and event streams](07_data_spec_eventstream_v1.md) for record/replay/export semantics.
4. [LeRobot interop](08_lerobot_interop_and_dataset_profile.md) when dataset interchange matters.
