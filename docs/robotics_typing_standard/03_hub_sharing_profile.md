# Hub Sharing Profile (v1)

This profile defines the minimum typing requirements for shareable robotics flows in GoldenRetriever Hub modules. Core runtime Hub mechanics live in the Retriever docs: https://openretriever-docs.pages.dev/ecosystem/.

## 1. Default Policy

Tuple input/output is allowed with constraints.

Required:
- each tuple element is an approved IO type,
- collision-safe qualified routing is resolvable,
- stamped boundary messages carry frame/time/source metadata where applicable.

## 2. Required Metadata for Robotics Boundary Types

For messages crossing process/network boundaries:
- timestamp (`stamp_ns`),
- frame id (`frame_id`),
- source id (`source`).

## 3. Compatibility Levels

- `profile=core`: allows unstamped internal-only messages.
- `profile=robotics_v1` (recommended): requires stamped boundary types and validation checks.
- `profile=strict_single_io`: optional strict mode for teams that disallow tuple IO.

## 4. Validation Checklist

Flow package should pass:
- tuple signature normalization checks,
- ambiguous unqualified access checks,
- arity/type checks for tuple outputs,
- stamped metadata presence checks for boundary ports.

## 5. Suggested Artifact Contents

A shareable flow bundle should include:
- declared input/output typing signature,
- routing/alias map for composite I/O,
- profile metadata (`core` or `robotics_v1`),
- minimal executable verification logs.
