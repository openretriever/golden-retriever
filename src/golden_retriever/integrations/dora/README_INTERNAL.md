# Dora Integration (Internal)

This package contains internal stubs and helpers for compiling Flow/Pipeline graphs to Dora dataflows.

- Frontend users never author Dora graphs or interact with Dora APIs directly.
- The backend executor is responsible for converting Flows to Dora nodes and wiring ports.
- Files here are used for internal development and experimentation only.

Key stubs
- `robot_io_node.py`: Reference node that owns a `RobotConnection`, receives `cmd`, and publishes `status`.
- `examples/`: Minimal producer/consumer stubs for development.
- `spec.py`: Internal `GraphSpec` and `NodeSpec` used by the backend executor to compile Flow graphs to Dora.

Coordination conventions (ports)
- `cmd`, `cmd_a`, `cmd_b`: command streams entering coordinator/arbiter nodes
- `status`: status stream from robot-io node
- `events`: optional event streams from monitors/safety nodes

Caveats
- APIs may change; no stability guarantees.
- Do not expose these scripts to end users or documentation.
