# 05 — FRP Coordination (Tutorial)

These examples focus on **coordination** patterns:
- multi-rate pipelines (fast sensing → slower estimation → slow planning)
- edge adapters (`Latest`, `Window`, `Hold`, …) defining how downstream nodes sample buffered history

## Run

```bash
# Multi-rate "robot system" toy demo
pixi run python -m examples.tutorial.05_frp_coordination.00_multirate_robot_system --backend dora --duration 5
```

