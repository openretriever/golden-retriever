# Robot Type Catalog and Semantics (v1)

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


Preferred module:
- `retriever_typing`

Pinned implementation module:
- `retriever_typing.v1`

Compatibility/registry access:
- `from retriever_typing import PoseStamped, SE3Pose`
- `from retriever_typing import get_type`
- `get_type("PoseStamped")`

## 1. Core Geometry Types

## `Vector3`
- Fields: `x`, `y`, `z`
- Unit depends on context:
  - position vector: meters
  - angular velocity: rad/s
  - force: newtons

## `Quaternion`
- Fields: `x`, `y`, `z`, `w`
- Semantics: right-handed rotation quaternion
- Constraint: norm should be approximately 1.0

## `SE3Pose`
- Fields: `position: Vector3`, `orientation: Quaternion`
- Semantics: rigid transform in 3D.

## 2. Motion and Force Types

## `Twist`
- Fields:
  - `linear: Vector3` (m/s)
  - `angular: Vector3` (rad/s)

## `Wrench`
- Fields:
  - `force: Vector3` (N)
  - `torque: Vector3` (N*m)

## 3. Joint Type

## `JointState`
- Fields:
  - `names: tuple[str, ...]`
  - `positions: tuple[float, ...]`
  - `velocities: tuple[float, ...]`
  - `efforts: tuple[float, ...]`
- Contract: all arrays align by index and length.

## 4. Stamped Wrappers

## `Header`
- Fields:
  - `stamp_ns: int`
  - `frame_id: str`
  - `source: str`

## `PoseStamped`
- Fields: `header`, `pose`

## `TwistStamped`
- Fields: `header`, `twist`

## `WrenchStamped`
- Fields: `header`, `wrench`

## 5. Validation Policy (v1)

Required checks at boundary nodes:
- non-empty `frame_id`,
- monotonic `stamp_ns` per source,
- quaternion near unit norm,
- aligned `JointState` array lengths.

## 6. Optional Extensions (v2+)

- Covariance-bearing types (`PoseWithCovariance`, `TwistWithCovariance`).
- Explicit unit wrapper metadata for non-SI systems.
- Frame graph provenance for transform lookup/debugging.
