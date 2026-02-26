# Motion Planning examples using RoboPlan

This folder contains some basic examples of motion planning using Retriever and [RoboPlan](https://github.com/open-planning/roboplan).

To set up these examples (supported on Linux only at the moment):

```bash
pixi install -e roboplan
```

You can then run the examples.

```bash
pixi run -e roboplan python examples/advanced/motion_planning/ik_example.py
pixi run -e roboplan python examples/advanced/motion_planning/motion_planning_example.py
```

NOTE: The first time you run these examples, the [Viser](https://github.com/nerfstudio-project/viser) visualizer may take a while to load.
