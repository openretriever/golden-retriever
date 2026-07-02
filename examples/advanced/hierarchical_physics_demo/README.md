# Hierarchical Physics Demo

Time-aware physics simulations with explicit clock → sim → viz layers.
Each flow logs to Rerun and prints periodic status to stdout.

Run:

```bash
pixi run -e twist2 python examples/advanced/hierarchical_physics_demo/double_pendulum.py --duration 6
pixi run -e twist2 python examples/advanced/hierarchical_physics_demo/three_body.py --duration 8
pixi run -e twist2 python examples/advanced/hierarchical_physics_demo/app.py --demo both --duration 8
```

Notes:
- Defaults use the Dora backend; override with `--backend multiprocessing`.
- Use `--no-rerun` to disable visualization.
- Use `--print-every 0` to disable stdout progress.
- Pipeline visualization summary (HTML path + ASCII graph) is logged to Rerun by default; disable with `--no-viz-html`.
- The interactive HTML auto-opens in your browser; disable with `--no-open-viz`.
