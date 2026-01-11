# Real-Time Hybrid Systems (Deadlines)

Hybrid, time-aware demos that couple continuous dynamics with discrete mode
switches and explicit deadline monitoring. Each demo uses multi-rate hierarchies
(mode @ low Hz, control @ mid Hz, sim @ high Hz) and `Latest()` adapters for
cross-rate sync. Defaults run on the Dora backend for parallel execution.

Run:

```bash
pixi run python examples/advanced/real_time_hybrid_systems/bouncing_ball_hybrid.py --duration 30
pixi run python examples/advanced/real_time_hybrid_systems/hybrid_deadline_throttle.py --duration 30
pixi run python examples/advanced/real_time_hybrid_systems/autopilot_mode_manager.py --duration 30
```

Notes:
- All demos default to wall-clock time; use `--fixed-dt` for deterministic dt.
- `--deadline-ms` and `--work-ms` let you induce deadline misses.
- Use `--no-rerun` to disable visualization.
- `bouncing_ball_hybrid.py` supports `--trail-len`, `--ground-width`, and `--ball-radius` for the 2D view.
- `hybrid_deadline_throttle.py` and `autopilot_mode_manager.py` log 2D ground/marker/trail visuals.
- Use `--trail-len`, `--ground-width`, and `--marker-radius` to tune their 2D views.
- Use `--no-invert-viz` to keep the simulation Y axis instead of the screen-friendly view.
- Use `--profile-scale` to stretch the time axis in the altitude profile view.
