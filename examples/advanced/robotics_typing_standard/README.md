# Robotics Typing Standard Demos

These examples accompany `docs/robotics_typing_standard/`.

## Files
- `type_catalog_demo.py`
  - quick runnable demo of stamped pose/twist/wrench + joint state.
- `compositional_contract_demo.py`
  - runnable demonstration of composite I/O routing and ambiguity behavior.

Canonical definitions live in `src/golden_retriever/robotics_typing/v1.py`.

## Run
```bash
python examples/advanced/robotics_typing_standard/type_catalog_demo.py
python examples/advanced/robotics_typing_standard/compositional_contract_demo.py
```

## What to look for
- boundary payloads include frame/time/source metadata,
- quaternion/joint-state validation checks,
- strict collision behavior:
  - unique unqualified fields work,
  - ambiguous unqualified fields raise,
  - qualified access is deterministic.
