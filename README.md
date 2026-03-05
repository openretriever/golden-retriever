# GoldenRetriever

**Retriever System Runtime & Advanced Integrations**

This repository contains the system-level implementations, robot integrations, and advanced research pipelines built on top of the generic `retriever` core.

## Components
- **`src/golden_retriever`**: Core system logic, robot drivers, environment wrappers.
- **`examples/advanced`**: Production-grade examples (VLA, MuJoCo, Real-Time Hybrid).
- **`examples/experimental`**: Research prototypes (Closed-Loop Planning, VLM Agents).
- **`docs/robotics_typing_standard`**: Robotics typing contract (`SE3`, `Twist`, `Wrench`, compositional I/O semantics).

## Setup
```bash
pixi install
pixi run app
```

## Robotics Typing Demos
```bash
pixi run demo-robotics-typing-catalog
pixi run demo-robotics-typing-contract
pixi run demo-robotics-typing-boundary
```

Canonical robotics typing API:
- `retriever_typing`
- `retriever_typing.v1` (pinned implementation path)
- `golden_retriever.types` (re-export + `get_type(...)`)
