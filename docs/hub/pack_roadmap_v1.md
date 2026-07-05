# Pack Maturity Guide

Golden is the applied reference layer for Retriever Hub packs. Not every useful example should become a pack immediately: a pack is a reusable boundary, while a source example can stay exploratory.

## What Is Loadable Now

| Pack surface | Status | Why it is safe to load |
| --- | --- | --- |
| Applied robotics type pack | Declared in `pyproject.toml` | Stable, import-safe, and useful across examples. |
| Arrow conversion helpers | Declared in `pyproject.toml` | Lightweight conversion checks can run without hardware or simulator dependencies. |

Run the current proof:

```bash
pixi run demo-golden-hub-pack
```

## Maturity Levels

| Level | Reader expectation |
| --- | --- |
| Source example | Useful pattern in the repo; read it after the promoted path works. |
| Promoted demo | Has a named Pixi command, expected output, and documented dependency level. |
| Hub-loadable pack | Import-safe, versioned, manifest-declared, smoke-tested, and useful outside this repo. |
| Optional integration | Requires a camera, model, simulator, GPU, robot, or external service and should stay clearly labeled. |

## Pack Readiness Checklist

A Golden example is ready to become a Retriever Hub pack when it has:

- import-safe top-level code,
- lightweight serializable construction config,
- no camera, robot, simulator, GPU, socket, model key, or file opening at import time,
- a named Pixi smoke command,
- a documented expected artifact,
- a clear dependency level,
- stable export names and versioning expectations.

## Good Candidate Shapes

These example shapes are good candidates once they meet the checklist:

| Shape | First smoke | Why it fits Hub |
| --- | --- | --- |
| Synthetic perception Flow | `pixi run -e golden-local demo-perception-detection-flow` | Deterministic, no hardware, typed perception payloads. |
| Belief updater Flow | `pixi run -e golden-local demo-memory-belief-flow` | Stateful partial-observability reference with a compact contract. |
| Caption-to-plan Flow | `pixi run -e golden-local demo-language-caption-plan` | Lightweight language payload reference; model-backed variants can remain optional. |
| Composable pipeline | `pixi run -e golden-local demo-composable-pipelines` | Demonstrates reusable pipeline boundaries and pipeline-as-Flow structure. |
| Graph visualization utility | `pixi run demo-pipeline-html-viz` | Useful inspection artifact when ownership and dependency boundaries are clear. |
| Mock simulator wrapper | `pixi run demo-robosuite-mock` | Shows environment-as-Flow and policy-as-Flow while keeping real simulator dependencies optional. |

## Keep As Source Examples Until Ready

Keep a lane source-level when it needs a stronger dependency story or a clearer public boundary: model-backed perception, simulator-heavy paths, hardware-bound examples, browser/operator surfaces, and design-pattern extracts without a named command and expected output.
