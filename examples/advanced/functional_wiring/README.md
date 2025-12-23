# Functional Wiring Examples

This folder demonstrates **Functional Reactive Programming (FRP)** style wiring patterns in Retriever.

## Overview

Retriever's wiring API is inspired by **Arrow combinators** from functional programming:

| FRP Concept | Retriever API | Description |
|-------------|---------------|-------------|
| Sequential (`>>>`) | `.then()` / `>>` | Chain flows: `a.then(b).then(c)` |
| Fan-In | `dst(src1, src2)` | Multiple sources → one destination |
| Fan-Out | `src >> (a & b)` | One source → multiple destinations (NEW!) |
| Sync Policy | `sync=Window(...)` | Control temporal alignment |

### Single-Expression Graph Building
```python
# Build entire graph in one line!
sensor >> (detector & logger & recorder)

# Equivalent to:
sensor.then(detector)
sensor.then(logger)
sensor.then(recorder)
```


## Examples

### 1. `../wiring_comparison.py` - Three Wiring Styles
Compares **Explicit**, **Fluent**, and **Functional** wiring:

```python
# Explicit
pipe.connect(source, fusion, map={"val": "values"})

# Fluent  
source.then(fusion, map={"val": "values"})

# Functional
fusion(source1, source2, source3)
```

Run: `pixi run demo-wiring-comparison`

### 2. `sync_policies.py` - Temporal Alignment
Demonstrates different sync adapters:

```python
# Window: Aggregate last 0.5s with mean
fusion(source, sync=Window(buffer_size=20, duration=0.5, agg="mean"))
```

### 3. `chaining.py` - Sequential Composition
Shows FRP-style chaining:

```python
# Build pipeline in one expression
source.then(processor).then(logger)
```

### 4. `fanout.py` - Signal Splitting
Demonstrates one source feeding multiple destinations:

```python
# Fan-out: source feeds both detector AND logger
source.then(detector)
source.then(logger)
```

## Quick Start

```bash
pixi run demo-wiring-comparison
pixi run demo-chaining
pixi run demo-fanout
pixi run demo-sync-policies
```

## Key Concepts

### Adapters (Sync Policies)
Control how data is sampled across different clock rates:

- **`Latest()`**: Most recent value (default)
- **`Window(duration, agg)`**: Aggregate over time window
- **`Hold(debounce)`**: Zero-order hold with debounce
- **`Events(duration)`**: Raw event stream access
- **`Exact(tolerance)`**: Timestamp-matching (for synchronized clocks)

### Context Management
Wiring automatically registers to the active pipeline:

```python
# Top-level scripts use default_pipeline() automatically
fusion(a, b)  # Registers to default

# Explicit pipeline context
with Pipeline("custom") as pipe:
    fusion(a, b)  # Registers to 'custom'
```
