# RoboSuite Lift Compatibility Entry Point

The maintained Lift implementation now lives in
[`../robocasa/robosuite_lift.py`](../robocasa/robosuite_lift.py), alongside the
RoboCasa simulator adapters and browser tooling. This package preserves the
original import and module paths for existing examples, tests, and links.

```bash
python -m examples.advanced.robosuite_lift.app --mode mock
```

New code may import from either path; both resolve to the same implementation.
