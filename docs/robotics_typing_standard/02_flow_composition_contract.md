# Flow Composition Contract (v1)

## 1. Supported Signatures

Equivalent accepted forms:
- `Flow[(A, B), C]`
- `Flow[tuple[A, B], C]`
- `Flow[A, (C, D)]`
- `Flow[(A, B), (C, D)]`

## 2. Input Routing Semantics

For composite input `(A, B)`, define:
- alias map: `A`, `B` (or deterministic suffixed aliases on conflict),
- unqualified map: `field -> [aliases]`,
- qualified map: `Alias.field -> (Alias, field)`.

Access rules:
- `inp.field` succeeds only when `field` maps to one alias.
- `inp.field` raises ambiguity when `field` maps to multiple aliases.
- `inp.A.field` / `inp.B.field` always valid when field exists.

## 3. Signal API Rules

- `_get_signal("field")`
  - succeeds if unique;
  - raises ambiguity if multi-source.
- `_get_signal("A.field")`
  - always explicit and valid if present.

Same for:
- `_set_signal(...)`
- `_has_signal(...)`

No implicit broadcast on ambiguous unqualified names.

## 4. Duplicate Type Name Policy

If two source types share the same class name, aliases are deterministic:
- `Name__1`, `Name__2`, ... by declaration order.

Qualified access uses aliases.

## 5. Output Routing Semantics

For output tuple `(C, D)`:
- result arity must match declared output arity,
- each tuple element must type-match its declared slot,
- downstream ports use normalized alias map identical to input policy.

## 6. Negative Cases

Must raise at runtime (and ideally validate ahead of execution):
- mixed tuple with `None` element in composite contract,
- ambiguous unqualified read/write/has,
- output tuple arity mismatch,
- output element type mismatch.
