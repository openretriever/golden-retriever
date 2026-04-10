# Robotics Typing Rollout Report

Date: 2026-02-27
Repo: `GoldenRetriever`

## Summary

This rollout established a concrete robotics typing surface in `GoldenRetriever` with:
- a canonical robotics typing package,
- direct import and registry lookup access,
- a medium-scope migration on high-impact modules,
- explicit serialization identifiers for robotics v1 payloads,
- a documented upstream carry-back plan for `retriever-mirror`.

## Supported API Surface

Preferred imports:

```python
from retriever_typing import PoseStamped, SE3Pose, JointState
```

Pinned/versioned imports:

```python
from retriever_typing.v1 import PoseStamped, SE3Pose, JointState
```

Registry lookup:

```python
from retriever_typing import get_type

PoseStamped = get_type("PoseStamped")
```

Decision:
- keep `v1` at the module boundary,
- do not put version suffixes into type names,
- prefer `retriever_typing` in normal user-facing examples.

## What Landed

### Canonical robotics type catalog

New package:
- `src/retriever_typing/v1.py`
- `src/retriever_typing/__init__.py`

Available payloads:
- `Header`
- `Vector3`
- `Quaternion`
- `SE3Pose`
- `PoseStamped`
- `Twist`
- `TwistStamped`
- `Wrench`
- `WrenchStamped`
- `JointState`

### Canonical package and registry discovery

Canonical classes are available through:
- direct imports from `retriever_typing`,
- pinned imports from `retriever_typing.v1`,
- registry lookup with `retriever_typing.get_type(...)`.

Current recommendation:
- prefer `retriever_typing` in public examples,
- keep `.v1` for pinned implementation references,
- use `get_type(...)` only for dynamic lookup paths.

### Migrated files

The following high-impact modules were migrated to the new typing surface:
- `src/golden_retriever/flows/control/robot_io.py`
- `src/golden_retriever/flows/control/safety.py`
- `src/golden_retriever/flows/vision/camera.py`
- `src/golden_retriever/flows/vision/depth.py`
- `src/golden_retriever/flows/vision/detection.py`
- `src/golden_retriever/flows/vision/visualization.py`
- `src/golden_retriever/pipelines/perception/detection.py`
- `src/golden_retriever/robots/spot/examples/connection_demo.py`

Within this migrated slice:
- `retriever.types.core_types` imports were removed,
- old spatial names `Pose3` and `Transform3` were removed.

### Serialization contract

`src/retriever_typing/conversions.py` supports stable robotics payload identifiers including:
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

## Validation

Validated commands:

```bash
python examples/advanced/robotics_typing_standard/type_catalog_demo.py
python examples/advanced/robotics_typing_standard/compositional_contract_demo.py
python examples/advanced/robotics_typing_standard/perception_to_control_boundary_demo.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_robotics_v1_validation.py \
  tests/test_robotics_v1_registry.py \
  tests/test_robotics_typing_public_surface.py \
  tests/test_compositional_collision_contract.py
```

Observed results:
- both demo scripts run successfully,
- the boundary demo shows stamped frame transitions and typed command serialization,
- registry lookup resolves canonical classes,
- robotics v1 validation tests pass,
- public package surface matches pinned `v1` and registry lookup,
- collision-contract tests pass,
- migrated files are free of `retriever.types.core_types`, `Pose3`, and `Transform3`.

## Explicit Non-Goals For This Wave

Not done yet:
- full repo-wide migration,
- direct code changes in `retriever-mirror`,
- broad runtime architecture refactors,
- Hub validator enforcement in the upstream runtime repo.

## Next Step

Carry the same package, bootstrap, conversion, and scoped migration strategy back into `retriever-mirror` in small commits using:
- `docs/robotics_typing_standard/05_mirror_upstream_patch_plan.md`
