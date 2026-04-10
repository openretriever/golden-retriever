# Webcam + Rerun Perception Demo

A real-time perception pipeline using a webcam (with mock fallback), open-vocabulary detection, segmentation, and Rerun visualization.

## Quick Start

```bash
pixi run -e torch demo-webcam-rerun
```

## Direct command

```bash
pixi run -e torch python examples/advanced/webcam_rerun/app.py --queries "cup,keyboard"
```

## Record + Replay with MCAP

```bash
pixi run -e torch python examples/advanced/webcam_rerun/record_replay.py record --steps 50
pixi run -e torch python examples/advanced/webcam_rerun/record_replay.py view
pixi run -e torch python examples/advanced/webcam_rerun/record_replay.py replay
```

## Notes

- The first model-backed run will download checkpoints.
- Use `--cleanup` on `app.py` if you want to remove downloaded model caches after the run.
