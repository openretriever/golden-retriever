# Closed-Loop Planning Example

Belief-space planning demo with VLM predicates and epistemic tracking.

## Quick Start

```bash
# High-level VLM planning (webcam + web UI)
pixi run -e llm demo-highlevel-planning

# Simple pipeline (heuristic planner)
pixi run demo-simple-pipeline

# Complete pipeline (A* planner)
pixi run demo-complete-pipeline

# RISE pipeline (Simulated Spot)
pixi run demo-rise-pipeline

# Spot pipeline (Real Robot)
# Requires SPOT_IP and BOSDYN credentials
pixi run demo-spot-pipeline
```

## Structure

```
closed_loop_planning/
├── pipelines/          # Execution pipelines
│   ├── simple.py
│   ├── complete.py
│   ├── demo_highlevel_planning.py
│   ├── demo_highlevel_planning_deprecated.py
│   ├── rise_pipeline.py
│   └── spot_pipeline.py
│
├── flows/              # Retriever Flows
│   ├── environment.py  # GridEnvironmentFlow
│   ├── rise_env.py     # RiseEnvironmentFlow (Sim)
│   ├── spot_env.py     # SpotEnvironmentFlow (Real)
│   ├── perception.py
│   ├── task_planner.py
│   └── ...
│
├── types/              # Data Types
│   ├── flow_types.py
│   ├── domain.py
│   └── ...
└── ...
```

## Rerun Viewer

- If port `9876` is already in use, `demo_highlevel_planning.py` will pick a free port and spawn a new viewer to avoid version conflicts.
- To reuse an existing viewer, pass `--rerun-port <port>` and/or `--rerun-no-spawn`.
- If you see Rerun decode errors, ensure the viewer and Python SDK come from the same pixi environment.
- The HTML plan graph auto-opens once per run by default; disable via `RETRIEVER_PLAN_HTML_AUTO_OPEN=0`.

## Key Concepts

- **BeliefState**: State + epistemic tracking (Known/Unknown)
- **VisualPredicate**: VLM-evaluated predicates with prompts
- **Operator**: STRIPS-style planning operator
- **TaskPlannerFlow**: A* planner as a Flow
