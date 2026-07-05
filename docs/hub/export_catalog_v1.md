# Golden Pack Export Catalog v1


This page is generated from the current public boundary in `pyproject.toml`: Golden exposes robot-facing payloads and conversion helpers through `[tool.retriever.module.exports]`, and Retriever Hub loads that manifest.

## Module declaration

```toml
[tool.retriever.module]
module = "retriever_typing"
min_retriever_version = "0.0.1"
```

## Applied robotics payloads

| Export | Source | Use |
| --- | --- | --- |
| `WorldState` | `retriever_typing.robotics_types:WorldState` | Object poses, robot pose, scene timestamp. |
| `RobotState` | `retriever_typing.robotics_types:RobotState` | Joint/pose/state snapshot for robot-facing examples. |
| `BeliefGraph` | `retriever_typing.robotics_types:BeliefGraph` | Belief and memory payload for partially observed tasks. |
| `Skill` | `retriever_typing.robotics_types:Skill` | Named skill with parameters and confidence. |
| `Plan` | `retriever_typing.robotics_types:Plan` | Skill sequence for simple plan examples. |
| `StructuredPlan` | `retriever_typing.robotics_types:StructuredPlan` | Richer plan payload. |
| `TaskGoal` | `retriever_typing.robotics_types:TaskGoal` | Goal object for task-conditioned examples. |
| `Trajectory` | `retriever_typing.robotics_types:Trajectory` | Motion/control trajectory payload. |
| `ExecutionStatus` | `retriever_typing.robotics_types:ExecutionStatus` | Monitor/progress/status payload. |

## Core-compatible action payloads

| Export | Source | Use |
| --- | --- | --- |
| `Action` | `retriever_typing.core_types:Action` | Action command payload used by demos and Arrow round-trip smoke. |
| `Command` | `retriever_typing.core_types:Command` | Command-style control payload. |
| `Status` | `retriever_typing.core_types:Status` | Generic status payload. |

## Conversion helpers

| Export | Source | Use |
| --- | --- | --- |
| `convert_to_arrow` | `retriever_typing.conversions:convert_to_arrow` | Convert supported payloads into Arrow objects. |
| `convert_from_arrow` | `retriever_typing.conversions:convert_from_arrow` | Restore supported payloads from Arrow objects. |

## Smoke command

```bash
pixi run demo-golden-hub-pack
```

Expected output includes:

```text
Golden pack exports: WorldState, BeliefGraph, Skill, Plan, Trajectory, convert_to_arrow, convert_from_arrow
Registry WorldState: ...WorldState
Constructed WorldState: ['cup']
Constructed Plan skills: ['pick']
Arrow round-trip: Action OK
Hub reference: hub.use("openretriever/golden-retriever:WorldState")
```

## Compatibility contract

The cross-version contract is the registered schema and serialization behavior, not Python class object identity. For a real application, pin one Golden ref per app or experiment and upgrade deliberately.
