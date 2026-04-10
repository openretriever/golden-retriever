# Robotics Typing Standard Demos

These examples accompany `docs/robotics_typing_standard/`.

## Files
- `type_catalog_demo.py`
  - quick runnable demo of stamped pose/twist/wrench + joint state.
- `compositional_contract_demo.py`
  - runnable demonstration of composite I/O routing and ambiguity behavior.
- `perception_to_control_boundary_demo.py`
  - representative stamped-boundary walkthrough: camera-frame target -> base-frame target -> typed control command -> serialization roundtrip.
- `data_spec_eventstream_demo.py`
  - deterministic event ordering + processing-time sampling profile.
- `multi_stream_join_demo.py`
  - event-time multi-stream joins (`exact`, `latest_before`, `window`) with lineage.
- `lerobot_bridge_demo.py`
  - canonical event-table row export and LeRobot mapping roundtrip.

Preferred import surface:
- `retriever_typing`

Pinned implementation path:
- `retriever_typing.v1`

## Run
```bash
python examples/advanced/robotics_typing_standard/type_catalog_demo.py
python examples/advanced/robotics_typing_standard/compositional_contract_demo.py
python examples/advanced/robotics_typing_standard/perception_to_control_boundary_demo.py
python examples/advanced/robotics_typing_standard/data_spec_eventstream_demo.py
python examples/advanced/robotics_typing_standard/multi_stream_join_demo.py
python examples/advanced/robotics_typing_standard/lerobot_bridge_demo.py
```

## What to look for
- boundary payloads include frame/time/source metadata,
- quaternion/joint-state validation checks,
- perception/control boundaries preserve frame transitions explicitly,
- typed payloads survive serialization with stable identity,
- deterministic event-time merge and join behavior across streams,
- processing-time sampling profile (`latest`, `hold`, `window_agg`) for compatibility,
- LeRobot mapping helpers for dataset interop,
- strict collision behavior:
  - unique unqualified fields work,
  - ambiguous unqualified fields raise,
  - qualified access is deterministic.
