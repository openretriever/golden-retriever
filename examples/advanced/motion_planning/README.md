# Motion Planning examples using RoboPlan

This folder contains some basic examples of motion planning using Retriever and [RoboPlan](https://github.com/open-planning/roboplan).

To set up these examples:

```bash
pixi install -e roboplan
```

The current `pixi.toml` exposes the `roboplan` environment on both `linux-64` and `osx-arm64`.

You can then run the examples.

```bash
pixi run -e roboplan demo-ik
pixi run -e roboplan demo-motion-track
```

NOTE: The first time you run these examples, the [Viser](https://github.com/nerfstudio-project/viser) visualizer may take a while to load.
