# Integrated Robotics Typing Standard v1

## 1. Problem

Robotics pipelines often pass values that look simple (`pose`, `velocity`, `force`) but are semantically incomplete unless we also track:
- reference frame (`base_link`, `map`, `camera_color_optical_frame`, ...),
- timestamp,
- units,
- source lineage.

Without this metadata in type contracts, composed flows silently mix incompatible signals.

## 2. Design Objectives

1. Keep authoring readable.
- `Flow[(Observation, Plan), (Action, Progress)]` should be valid and explicit.

2. Make signal meaning first-class.
- Typed payloads include frame/timestamp where needed.

3. Make compositional routing deterministic.
- Unique unqualified field access is allowed.
- Ambiguous unqualified access must raise.
- Qualified access (`A.field`) is always valid.

4. Prepare for shareable flow artifacts.
- Hub/registry-ready profile includes minimal metadata requirements.

## 3. Canonical Type Families

v1 standardizes these payload families:
- Geometry: `Vector3`, `Quaternion`, `SE3Pose`.
- Motion: `Twist`, `Wrench`.
- Joint-level: `JointState`.
- Stamped wrappers: `PoseStamped`, `TwistStamped`, `WrenchStamped`.

Stamped wrappers carry:
- timestamp (`stamp_ns`),
- frame id (`frame_id`),
- source id (`source`).

Canonical access surfaces:
- preferred import: `from retriever_typing import PoseStamped`
- pinned import: `from retriever_typing.v1 import PoseStamped`
- convenience import: `from retriever_typing import PoseStamped`
- registry lookup: `from retriever_typing import get_type`

## 4. Compositional Flow Typing Model

Supported signature forms:
- `Flow[(A, B), C]`
- `Flow[tuple[A, B], C]`
- `Flow[A, (C, D)]`
- `Flow[(A, B), (C, D)]`

Normalized semantics are the same regardless of tuple-literal vs `tuple[...]` style.

## 5. Collision Contract

Given input composite `(A, B)`:
- If `field` exists in only one source: `inp.field` is valid.
- If `field` exists in both sources: `inp.field` must raise ambiguity.
- Qualified reads/writes are always valid:
  - `inp.A.field`, `inp.B.field`
  - `_get_signal("A.field")`, `_set_signal("B.field", value)`

No implicit broadcast writes are allowed for ambiguous names.

## 6. Recommended Authoring Pattern

- Keep flow constructors light (`__init__` stores config).
- Keep heavy runtime setup in `init` / worker-side hooks.
- Use stamped payloads at boundaries; internal nodes can downcast to leaner types when safe.

## 7. Expected Outcomes

- Safer cross-team composition.
- Fewer frame/unit mistakes.
- Cleaner migration path toward strict shareable-flow typing checks.

## 8. Boundary Walkthrough

Representative example flow:
1. perception emits `PoseStamped` in `camera_color_optical_frame`,
2. normalization emits `PoseStamped` in `base_link`,
3. controller emits `TwistStamped` in `base_link`,
4. logging/replay path serializes the typed command with stable type identity.

Reference demo:
- `examples/advanced/robotics_typing_standard/perception_to_control_boundary_demo.py`
