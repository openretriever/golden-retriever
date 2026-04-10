# Robotics Typing Standard

This folder contains the public GoldenRetriever notes for robotics-focused payload types, compositional flow contracts, and event/data profiles.

## Recommended reading order

- `00_integrated_robotics_typing_standard_v1.md`: end-to-end narrative and usage model.
- `01_type_catalog_and_semantics.md`: canonical type catalog (`SE3Pose`, `Twist`, `Wrench`, `JointState`, stamped variants).
- `02_flow_composition_contract.md`: compositional signatures like `Flow[(A, B), C]` and collision semantics.
- `03_hub_sharing_profile.md`: minimum metadata/profile for shareable flows.
- `06_rollout_report_2026-02-27.md`: rollout results and validation summary.
- `07_data_spec_eventstream_v1.md`: event/data contracts and multi-stream semantics.
- `08_lerobot_interop_and_dataset_profile.md`: dataset manifest and LeRobot mapping profile.

## Running the demos

For runnable examples in this repo, start with `examples/advanced/robotics_typing_standard/README.md`.

## Historical note

- `05_mirror_upstream_patch_plan.md` is kept only as an archival integration-history note.
