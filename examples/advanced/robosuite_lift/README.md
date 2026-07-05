# RoboSuite Lift Demo

This is the smallest GoldenRetriever robosuite lane. It shows how to wrap a simulator environment and a simple scripted policy as Retriever `Flow` objects with different clocks.

The default path is mock-safe, so docs and smoke tests do not require robosuite:

```bash
pixi run demo-robosuite-mock
```

For a real robosuite run from a source checkout, install the optional dependency without asking pip to resolve the base package dependencies, then run:

```bash
# --no-deps: the Pixi environment supplies base deps; install robosuite explicitly.
pixi run python -m pip install --no-deps -e ".[robosuite]" robosuite
pixi run demo-robosuite-lift
```

Drop `--no-deps` only after the core runtime package is published and resolvable from PyPI in the target environment.

## Graph

```text
LiftEnvFlow @ 20 Hz ──Latest──▶ HeuristicLiftPolicy @ 5 Hz
       ▲                                      │
       └──────────────Latest─────────────────┘
       └──Latest──▶ LiftPrinter @ Trigger("step")
```

## What To Observe

- The simulator/environment wrapper and policy are separate Flows.
- `Latest()` makes the slow policy consume the newest simulator state and the simulator consume the newest policy command.
- The mock mode demonstrates the graph contract without robosuite.
- The real mode uses robosuite's `Lift` task with a `Panda` robot when the optional dependency is available.

This is intentionally a smoke demo, not a trained manipulation policy.
