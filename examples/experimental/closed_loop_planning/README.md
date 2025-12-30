# Closed-Loop Planning Example

Belief-space planning demo with VLM predicates and epistemic tracking.

## Quick Start

```bash
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

## Key Concepts

- **BeliefState**: State + epistemic tracking (Known/Unknown)
- **VisualPredicate**: VLM-evaluated predicates with prompts
- **Operator**: STRIPS-style planning operator
- **TaskPlannerFlow**: A* planner as a Flow
