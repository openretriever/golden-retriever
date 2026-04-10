# JAX (Flax) Async Examples

JAX / Flax examples for async inference and split-learning style training.

## Current status

This repo does not define a dedicated Pixi `jax` environment today. Run these examples from a manual JAX-enabled environment instead.

```bash
python examples/advanced/jax_cuda_async/inference.py
python examples/advanced/jax_cuda_async/train.py
```

## Files

- `inference.py`: wrap a Flax module and run inference inside a Retriever pipeline.
- `train.py`: async split-learning style training example.
- `DESIGN.md`: implementation notes and constraints.
