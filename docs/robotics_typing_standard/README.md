# Robotics Typing Standard (Golden Retriever)

This folder defines the working typing standard for robotics-focused flows in this repository.

## Why this exists
- Robotics data is not just tensor shape: frame, timestamp, unit, and lineage are part of correctness.
- Shared flows (future Hub use) need deterministic, machine-checkable contracts.
- Compositional flow signatures (`Flow[(A, B), C]`) require strict collision semantics.

## Documents
- `00_integrated_robotics_typing_standard_v1.md`
  - End-to-end narrative and usage model.
- `01_type_catalog_and_semantics.md`
  - Canonical type catalog (`SE3Pose`, `Twist`, `Wrench`, `JointState`, stamped variants).
- `02_flow_composition_contract.md`
  - `Flow[(A, B), C]` routing and ambiguity behavior.
- `03_hub_sharing_profile.md`
  - Minimum metadata/profile for shareable flows.
- `05_mirror_upstream_patch_plan.md`
  - Mirror-ready upstream carry-back patch map and dependency order.
- `06_rollout_report_2026-02-27.md`
  - Public rollout summary, validation results, and remaining follow-up.

## Canonical API
- `retriever_typing` (preferred import surface)
- `retriever_typing.v1` (pinned implementation/version path)
- `retriever_typing.get_type(...)` (registry lookup)

## Scope
This is the design source of truth for now in `GoldenRetriever`. Runtime enforcement can be added incrementally.
