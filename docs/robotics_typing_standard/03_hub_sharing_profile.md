# Hub Sharing Profile

Golden examples can become Retriever Hub packs only after their imports, type contracts, dependency story, and smoke checks are stable. This page defines the minimum profile for robot-facing flows and payload packs.

## Default Policy

Tuple input/output is allowed when the routing contract is explicit and each tuple element is an approved I/O payload.

Required for robot-facing boundaries:

- stamped metadata where frame, time, or source matters,
- deterministic qualified routing for composite I/O,
- small smoke command that can run without hardware,
- dependency tier label for optional simulator, camera, model, or robot integrations.

## Compatibility Levels

| Level | Meaning | Use |
| --- | --- | --- |
| `core` | Internal-only payloads with no robot boundary guarantees. | Small local examples. |
| `robotics_v1` | Stamped boundary payloads plus validation checks. | Default for Golden robot-facing packs. |
| `strict_single_io` | Disallows tuple I/O for teams that require one input and one output envelope. | Conservative production integrations. |

## Suggested Pack Contents

A shareable robotics pack should include:

- declared input/output type signatures,
- routing or alias map for composite I/O,
- profile metadata such as `core` or `robotics_v1`,
- import examples for both convenience and pinned paths,
- smoke command and expected terminal output,
- dependency tier notes for optional integrations.

## Promotion Checklist

| Check | Why |
| --- | --- |
| `pixi run demo-golden-hub-pack` passes | The pack is visible through the Hub manifest and registry. |
| Type imports work from a clean environment | Users can load the pack without repo-specific path tricks. |
| Public docs name the first command | A new user knows how to verify the pack. |
| Optional dependencies are labeled | Camera/simulator/model paths do not block the mock-safe route. |
| Public surface check passes | The docs, tasks, and expected artifacts stay aligned. |
