# Robot Type Catalog

<div class="gr-route-pills gr-route-pills-inline">
  <a href="/">Golden overview</a>
  <a href="/examples/">Examples</a>
  <a href="/hub/">Hub packs</a>
  <a href="/robotics_typing_standard/">Robot type packs</a>
  <a href="/llms.txt">Agent map</a>
</div>

Golden's type pack gives robot examples a shared vocabulary: poses, twists, wrenches, joint state, timestamps, plans, skills, and execution status. Use these types when examples need to agree on payload meaning without pulling in a simulator or ROS message package.

## Quick Import

```python
from retriever_typing import PoseStamped, SE3Pose, get_type

pose_type = get_type("PoseStamped")
```

| Surface | Value |
| --- | --- |
| Preferred module | `retriever_typing` |
| Pinned implementation | `retriever_typing.v1` |
| Lookup style | `get_type("PoseStamped")` |

## Geometry Types

| Type | Fields | Meaning |
| --- | --- | --- |
| `Vector3` | `x`, `y`, `z` | Generic 3D vector. Units depend on context: meters for position, radians/sec for angular velocity, newtons for force. |
| `Quaternion` | `x`, `y`, `z`, `w` | Right-handed rotation quaternion. Boundary checks should keep the norm near 1.0. |
| `SE3Pose` | `position: Vector3`, `orientation: Quaternion` | Rigid 3D pose. Use with a stamped wrapper when frame or time matters. |

## Motion and Force Types

| Type | Fields | Meaning |
| --- | --- | --- |
| `Twist` | `linear: Vector3`, `angular: Vector3` | Spatial velocity: linear m/s and angular rad/s. |
| `Wrench` | `force: Vector3`, `torque: Vector3` | Spatial force: newtons and newton-meters. |

## Joint State

| Type | Fields | Contract |
| --- | --- | --- |
| `JointState` | `names`, `positions`, `velocities`, `efforts` | Arrays align by index and have the same length. |

Use `JointState` for compact examples where a full robot description is unnecessary but joint-level state still needs a stable schema.

## Stamped Wrappers

| Type | Fields | Use when |
| --- | --- | --- |
| `Header` | `stamp_ns`, `frame_id`, `source` | A value needs time, frame, and provenance. |
| `PoseStamped` | `header`, `pose` | A pose belongs to a frame at a specific timestamp. |
| `TwistStamped` | `header`, `twist` | A velocity estimate belongs to a frame and source. |
| `WrenchStamped` | `header`, `wrench` | A force/torque estimate belongs to a frame and source. |

## Validation Policy

Boundary nodes should reject or normalize values before passing them into reusable flows:

- `frame_id` is non-empty when a frame matters.
- `stamp_ns` is monotonic per source where ordering matters.
- `Quaternion` norm is close to 1.0.
- `JointState` arrays have aligned lengths.

## Future Extensions

The v1 catalog intentionally stays small. Later packs can add covariance-bearing poses/twists, explicit unit metadata for non-SI systems, and frame-graph provenance for transform debugging.
