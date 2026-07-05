# Robot Type Catalog

Golden's type pack gives robot examples a shared vocabulary: spatial values, stamped wrappers, world state, belief, plans, trajectories, and execution status. Use these types when examples need to agree on payload meaning without pulling in ROS messages or a simulator-specific API.

## Quick Import

```python
from retriever_typing import PoseStamped, SE3Pose, get_type
from retriever_typing.robotics_types import WorldState, Plan, ExecutionStatus

pose_type = get_type("PoseStamped")
```

```bash
pixi run demo-robotics-typing-catalog
```

## Spatial Values

These names are re-exported by `retriever_typing` from the canonical Retriever runtime spatial types, so a Golden example and a core runtime component agree on the class identity.

| Type | Fields | Use |
| --- | --- | --- |
| `Vector3` | `x`, `y`, `z` | Generic 3D vector. Units come from context. |
| `Quaternion` | `x`, `y`, `z`, `w` | Right-handed rotation quaternion. Validate norm near 1.0 at boundaries. |
| `SE3Pose` | `position`, `orientation` | Rigid 3D pose. Wrap it when frame or time matters. |
| `Twist` | `linear`, `angular` | Spatial velocity. |
| `Wrench` | `force`, `torque` | Spatial force/torque. |
| `JointState` | `names`, `positions`, `velocities`, `efforts` | Joint arrays aligned by index. |

## Stamped Boundaries

Use stamped wrappers at graph boundaries, dataset boundaries, process boundaries, and robot interfaces.

| Type | Adds | Use when |
| --- | --- | --- |
| `Header` | `stamp_ns`, `frame_id`, `source` | A payload needs time, frame, and provenance. |
| `PoseStamped` | `Header` + `SE3Pose` | A pose belongs to a frame at a specific time. |
| `TwistStamped` | `Header` + `Twist` | A velocity estimate belongs to a frame and source. |
| `WrenchStamped` | `Header` + `Wrench` | A force/torque estimate belongs to a frame and source. |

## Robot Task Payloads

| Type | What it represents | Typical producer/consumer |
| --- | --- | --- |
| `WorldState` | Object poses, robot pose, optional timestamp. | Perception, memory, planner. |
| `RobotState` | Robot pose tuple, held objects, battery, last error. | Robot wrapper, monitor, policy. |
| `BeliefGraph` | Probabilistic graph over symbolic scene nodes. | Memory, grounding, planner. |
| `Skill` | Named robot capability with params and confidence. | Planner, skill selector. |
| `Plan` / `StructuredPlan` | Sequence of skills or typed steps. | Language planner, task planner. |
| `TaskGoal` | High-level task plus affordances and success criteria. | User interface, planner. |
| `Trajectory` | Time-indexed sequence of poses. | Motion planner, controller. |
| `ExecutionStatus` | Progress, terminal state, and optional error message. | Controller, monitor, logger. |

## Validation Policy

Boundary nodes should reject or normalize values before passing them into reusable flows:

- `frame_id` is non-empty when a frame matters.
- `stamp_ns` is monotonic per source where ordering matters.
- `Quaternion` norm is close to 1.0.
- `JointState` arrays have aligned lengths.
- `ExecutionStatus.progress`, when present, stays in `[0, 1]`.

## Keep The Catalog Small

The first catalog is intentionally compact. Add covariance-bearing poses, richer frame graphs, or domain-specific robot schemas as separate packs only after the base payloads are not enough.
