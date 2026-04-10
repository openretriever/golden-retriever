# Native Controller Example: Robot IK Solver

This example shows one Retriever pipeline with interchangeable IK implementations in Python, Rust, and C++.

## Quick Start

```bash
# Python baseline
pixi run binding-controller-python  # runs for 5 seconds
```

## Rust backend

```bash
cd examples/advanced/native_controller
cargo build --release
cd ../../..
pixi run binding-controller-rust  # runs for 5 seconds
```

## C++ backend

```bash
cmake -S examples/advanced/native_controller -B examples/advanced/native_controller/build
cmake --build examples/advanced/native_controller/build --config Release
pixi run binding-controller-cpp  # runs for 5 seconds
```

## Notes

- The `--backend` flag in `app.py` selects the IK implementation (`python`, `rust`, `cpp`).
- The Retriever runtime backend used by this example remains the pipeline runtime configured inside `app.py`.
