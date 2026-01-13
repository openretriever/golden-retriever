# BEHAVIOR-1K Experiment

This directory contains experiments using [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) and [OmniGibson](https://github.com/StanfordVL/OmniGibson).

## Requirements

*   **Operating System**: Linux (Ubuntu 20.04+ recommended)
*   **GPU**: NVIDIA RTX 2070+ (8GB+ VRAM) with appropriate drivers
*   **Project**: This directory is a standalone Pixi project.

## Running

Navigate to this directory and use `pixi` commands directly:

```bash
```bash
cd experiments/behavior_1k
pixi install
pixi run install-simulation  # First time setup (UV pip) - Linux Only
pixi run run-comet           # Runs solutions/comet.py
```

## Solutions

*   **Comet**: `solutions/comet.py` - OpenPI VLA inference using `Retriever` pipeline.
    *   Auto-downloads checkpoint from HuggingFace.
    *   Uses shared `OmniGibsonEnv`.

## Structure
*   `common/`: Shared Retriever Flows (e.g. `OmniGibsonEnv`).
*   `solutions/`: Specific solution pipelines.

## Setup

Dependencies are defined in `experiments/behavior_1k/pixi.toml` and are restricted to `linux-64` only.
