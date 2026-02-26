# Retriever Distribution

Pre-built wheel + pixi environment for running Retriever experiments without a local source build.

## Quick Start (Recommended: Pixi)

1. Install Pixi if you don't have it:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. Install the environment:
   ```bash
   cd retriever_dist
   pixi install
   ```

3. Verify the install:
   ```bash
   pixi run python -m examples.tutorial.011_debug_stepper --steps 5
   ```

4. Enter an interactive shell:
   ```bash
   pixi shell
   python -c "import retriever; print(retriever.__version__)"
   ```

## Alternative: pip

```bash
pip install install/retriever-0.0.0-py3-none-any.whl[dora,demo]
```

You will need to install additional deps (numpy, dora-rs, etc.) manually.

## Running Examples

All tutorial examples are in `examples/tutorial/`. Run them with:

```bash
# Stepper debugger (no external deps)
pixi run python -m examples.tutorial.011_debug_stepper

# Dora-based perception demo (requires webcam)
pixi run demo-webcam-detection

# Request-response pattern
pixi run python -m examples.tutorial.010_request_response

# Closed-loop environment
pixi run python -m examples.tutorial.016_closed_loop_env
```

## Environment Features

The default pixi environment includes:
- `retriever[dora,demo,web,recording]` from the bundled wheel
- `dora-rs` + `dora-rs-cli` for the dora backend
- `rerun-sdk` for visualization
- `google-genai` for VLM support

Optional features (pass `--environment` to pixi):
- `torch` — PyTorch + Transformers + OpenPI

## Wheel Contents

The wheel installs the `retriever` package with these extras available:
- `dora` — Dora dataflow backend
- `demo` — OpenCV, NumPy for perception demos
- `web` — FastAPI + uvicorn for command interfaces
- `recording` — MCAP recording support
