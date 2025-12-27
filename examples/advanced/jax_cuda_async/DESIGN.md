# Design Notes: JAX Integration

## Overview

The JAX integration has been refactored to align with the framework's unified **Wrapper Pattern**. This replaces earlier ad-hoc JAX support with a canonical implementation that supports `dora` zero-copy data passing and functional state management.

## Key Components

### 1. `JaxFlow` Internal Wrapper
*   **Role**: Wraps JAX/Flax modules in a `Flow`.
*   **State Management**: Handles the functional nature of JAX.
    *   Maintains `self.params` stateful storage.
    *   Compiles `self._apply_fn` using `jax.jit`.
    *   Initializes parameters lazily if `sample_input` is provided.

### 2. `from_jax` Factory
*   **Role**: Public entry point.
*   **Detection**: `Wrapper(obj)` automatically detects objects with `__module__` containing "jax" or "flax" and delegates to `from_jax`.
*   **Input**: Accepts a `flax.linen.Module` instance.

### 3. `JaxIO` & Zero-Copy
*   **Type**: `JaxIO` (dataclass)
*   **Protocol**: Implements `__arrow_array__` (optional support in future) or standard pickle serialization for now.
*   **Design**: Intended to facilitate zero-copy transfer of arrays between nodes when using the `dora` backend (via `pyarrow`).
