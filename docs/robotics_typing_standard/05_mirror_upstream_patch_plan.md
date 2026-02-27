# Mirror Upstream Patch Plan (Golden -> Mirror)

This document defines the exact carry-back patch bundle for `retriever-mirror`.

## Scope
- Golden-first implementation done here.
- No direct edits to `retriever-mirror` in this wave.
- `examples/tutorial_mirror` remains subtree-managed and read-only.

## Patch Bundle Order

1. Canonical typing package
- Add `src/retriever/robotics_typing/__init__.py`
- Add `src/retriever/robotics_typing/v1.py`
- Keep v1 type names and validator semantics identical.

2. Dual-surface exports
- Update `src/retriever/types/__init__.py`
- Add v1 symbols (`Header`, `Vector3`, `Quaternion`, `SE3Pose`, `PoseStamped`, `Twist`, `TwistStamped`, `Wrench`, `WrenchStamped`, `JointState`)
- Preserve existing legacy exports.

3. Registry bootstrap
- Update `src/retriever/types/registry.py`
- Add built-in bootstrap imports:
  - `retriever.types.core_types`
  - `retriever.types.vision_types`
  - `retriever.types.robotics_types`
  - `retriever.robotics_typing.v1`
- Ensure `get_type("SE3Pose")` works without user pre-imports.

4. Conversion contract
- Update `src/retriever/types/conversions.py`
- Add explicit v1 serializers/deserializers with stable IDs:
  - `robotics.v1.Header`
  - `robotics.v1.Vector3`
  - `robotics.v1.Quaternion`
  - `robotics.v1.SE3Pose`
  - `robotics.v1.PoseStamped`
  - `robotics.v1.Twist`
  - `robotics.v1.TwistStamped`
  - `robotics.v1.Wrench`
  - `robotics.v1.WrenchStamped`
  - `robotics.v1.JointState`
- Retire `Pose3` / `Transform3` conversion branches in migrated path.

5. High-impact migration slice
- Update flow and pipeline modules to remove `retriever.types.core_types` imports in migrated surfaces.
- Replace spatial boundary payloads with stamped v1 payloads where applicable.

6. Docs and demos
- Update robotics typing docs to canonical path `retriever.robotics_typing.v1`.
- Update advanced typing demos to import canonical module and include registry lookup example.

## Candidate Mirror Files

- `src/retriever/robotics_typing/__init__.py` (new)
- `src/retriever/robotics_typing/v1.py` (new)
- `src/retriever/types/__init__.py`
- `src/retriever/types/registry.py`
- `src/retriever/types/conversions.py`
- `src/retriever/flows/control/robot_io.py`
- `src/retriever/flows/control/safety.py`
- `src/retriever/flows/vision/depth.py`
- `src/retriever/flows/vision/camera.py`
- `src/retriever/flows/vision/detection.py`
- `src/retriever/flows/vision/visualization.py`
- `src/retriever/pipelines/perception/detection.py`
- `src/retriever/robots/spot/examples/connection_demo.py`
- `examples/advanced/robotics_typing_standard/type_catalog_demo.py`
- `examples/advanced/robotics_typing_standard/compositional_contract_demo.py`
- `docs/robotics_typing_standard/README.md`
- `docs/robotics_typing_standard/00_integrated_robotics_typing_standard_v1.md`
- `docs/robotics_typing_standard/01_type_catalog_and_semantics.md`

## Upstream Validation Checklist

1. `python examples/advanced/robotics_typing_standard/type_catalog_demo.py`
2. `python examples/advanced/robotics_typing_standard/compositional_contract_demo.py`
3. `pytest -q tests/test_robotics_v1_validation.py`
4. `pytest -q tests/test_robotics_v1_registry.py`
5. `pytest -q tests/test_compositional_collision_contract.py`
6. Verify no migrated file imports `retriever.types.core_types`.
7. Verify no migrated file references `Pose3` or `Transform3`.

## Rollback

- Roll back in reverse patch order.
- Keep canonical typing package files (`robotics_typing/v1.py`) intact unless serialization or naming regression is observed.
