# 03 — State Management (Tutorial)

These examples show how to manage **stateful** behavior inside Retriever flows.

Key ideas:
- A `Flow` can maintain internal state (like a controller, filter, or tracker).
- For **debugging**, use the in-process stepper so you can set breakpoints inside `Flow.run()`.
- `Flow.reset()` is the hook to reset internal state between runs.
- `Eff` can be used as a pure state-transition helper inside a Flow (state remains internal).

## Run

```bash
pixi run python -m examples.tutorial.03_state_management.00_stateful_flow_reset
```

## Examples (canonical)

- `00_stateful_flow_reset.py`: minimal state + reset + stepper-friendly debugging.
- `01_belief_updater.py`: belief update with implicit internal state.
- `02_stateful_composition.py`: multi-flow stateful composition (nav + battery).

## Legacy (archived in this folder)

- `legacy/01_eff_basics.py`
- `legacy/02_robotics_state.py`
- `legacy/03_state_intro.py`
- `legacy/04_immutable_state.py`
