# Flow Composition Contract

Reusable robot graphs often need more than one input: state, goal, detections, plan, controller feedback, or a simulator observation. This page defines the Golden contract for composing those typed payloads without relying on ambiguous field names.

## Runnable Check

```bash
pixi run demo-robotics-typing-contract
```

Expected result: unique field access succeeds, ambiguous unqualified access raises, and qualified access remains deterministic.

## Supported Signatures

Retriever Flow authoring should stay readable. These forms normalize to the same composite I/O model:

```python
Flow[(A, B), C]
Flow[tuple[A, B], C]
Flow[A, (C, D)]
Flow[(A, B), (C, D)]
```

## Input Routing Rules

For a composite input `(A, B)`, the runtime view has two maps:

| Map | Example | Meaning |
| --- | --- | --- |
| Alias map | `A`, `B` | Stable names for the input payloads. |
| Unqualified map | `pose -> [A]` | Short access is allowed only when one source owns the field. |
| Qualified map | `A.pose` | Explicit access is always deterministic when the field exists. |

Rules:

- `inp.field` succeeds only when exactly one input source has `field`.
- `inp.field` raises ambiguity when multiple sources have that field.
- `inp.A.field` and `inp.B.field` stay valid when the field exists.
- `_get_signal("A.field")`, `_set_signal("A.field", value)`, and `_has_signal("A.field")` follow the same explicit rule.

## Output Routing Rules

For output tuple `(C, D)`:

- result arity must match the declared output arity,
- each tuple element must match its declared slot,
- downstream ports use the same normalized alias policy.

## Why This Matters

The failure mode is common in robot systems: camera perception, state estimation, and simulation can all expose something called `pose`. Golden's contract makes the simple case short and the ambiguous case explicit instead of silently choosing one source.

## Negative Cases

A public pack should fail early for:

- ambiguous unqualified reads, writes, or existence checks,
- output tuple arity mismatch,
- output element type mismatch,
- mixed composite contracts that include `None` as a payload element.
