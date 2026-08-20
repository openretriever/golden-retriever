# RoboCasa Demonstration Replay

This example connects Retriever to an actual RoboCasa simulator. Recorded human
actions are emitted by one `Flow`, consumed by a RoboCasa physics `Flow`, and
published as typed observations for stdout or Rerun.

```text
DemoActionSource @ 20 Hz
          |
        Latest
          v
RoboCasaSimulator @ 20 Hz ---> ObservationPrinter @ Trigger
          |
          +-- native MuJoCo viewer, or
          +-- live browser scene through mjviser, or
          +-- offscreen camera + telemetry in Rerun
```

The layers have separate jobs: RoboCasa defines kitchen tasks and datasets;
RoboSuite composes the robot, objects, arena, and controllers; MuJoCo advances
physics; Retriever connects action and observation Flows; mjviser displays the
live MuJoCo model and data in a browser. Rerun remains the better view for
camera frames, Flow execution, scalar telemetry, and saved traces.

## 1. Mock-safe contract

Run this first. It needs no simulator, assets, camera, or GUI:

```bash
retriever run demo-robocasa-mock
```

Expected output includes deterministic progress and a successful final step:

```text
[mock step=0000] progress=0.0% reward=0.000 success=False
[mock step=0004] progress=36.4% reward=0.364 success=False
[mock step=0008] progress=72.7% reward=0.727 success=False
[mock step=0011] progress=100.0% reward=1.000 success=True
```

## 2. Verify the RoboSuite base layer

RoboCasa builds on RoboSuite. This family keeps a small mock-first Lift loop
next to the RoboCasa adapter so the lower-level environment and policy wiring
can be checked independently:

```bash
retriever run demo-robosuite-mock
```

After installing the optional RoboSuite dependency, run the same Flow graph
against its real `Lift` task:

```bash
pixi run demo-robosuite-lift
```

```text
LiftEnvFlow @ 20 Hz --Latest--> HeuristicLiftPolicy @ 5 Hz
       ^                                      |
       +----------------Latest----------------+
       +--Latest--> LiftPrinter @ Trigger("step")
```

The implementation lives in [`robosuite_lift.py`](robosuite_lift.py). The
lower-level [`robosuite_inspection.py`](robosuite_inspection.py) script exposes
the underlying MuJoCo model and native viewer directly:

```bash
mjpython examples/advanced/robocasa/robosuite_inspection.py
```

That inspection script was originally contributed by Sebastian Castro and is
kept as a separate entry point within this family.

## 3. Install the real RoboCasa lane

RoboCasa currently needs source checkouts plus its kitchen assets. Keep those
large dependencies outside Golden Retriever and install them editable into the
environment used to run this example. Follow the upstream RoboCasa setup, then
download one human demonstration dataset:

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
python -m robocasa.scripts.download_datasets \
  --tasks TurnOnMicrowave --split pretrain --source human
```

The example lazily imports RoboCasa, so the mock path and normal Golden
Retriever tests do not require this stack.

## 4. Run a real replay

Headless physics, suitable for Linux workers:

```bash
python -m examples.advanced.robocasa.app \
  --mode robocasa --task TurnOnMicrowave --seconds 45
```

`--seconds` includes environment construction. Budget roughly 15 seconds for
the first MJCF and asset initialization on a laptop before replay time begins.

Native MuJoCo viewer on macOS:

```bash
mjpython -m examples.advanced.robocasa.app \
  --mode robocasa --task TurnOnMicrowave --seconds 45 --viewer
```

Rerun camera frames, flow execution, and telemetry:

```bash
python -m examples.advanced.robocasa.app \
  --mode robocasa --task TurnOnMicrowave --seconds 45 --visualize rerun
```

Live browser scene from the same `MjModel` and `MjData` used by RoboCasa:

```bash
python -m examples.advanced.robocasa.app \
  --mode robocasa --task TurnOnMicrowave --seconds 60 --visualize mjviser
```

Open `http://localhost:8085`. This is a zero-copy view of the running
simulator, not a separately stepped WebAssembly scene or an exported XML with
default joint state. Install `mjviser` in the configured RoboCasa environment;
the external simulator environment owns the tested MuJoCo compatibility override.

The left **Retriever replay** panel controls the actual recorded-action Flow:
pause or resume the replay, advance one action while paused, restart the
episode, or slow execution for inspection. Its **Graph** tab renders a live
HTML node-and-edge view of the
`DemoActionSource -> RoboCasaSimulator -> ObservationPrinter` handoff. Restart
resets MuJoCo in place and increments the displayed replay cycle. New browser
clients open in the unobstructed agent-facing view; use the **Camera** control
to switch to a tracked third-person robot view or overhead overview. mjviser's
**Track camera** toggle changes between following the robot and a static world
frame.

Keep task selection one level above this run view. A demo launcher can start a
fresh Retriever session for `PrepareCoffee`, `TurnOnMicrowave`, or another
installed dataset; the viewer stays bound to one task because switching tasks
can replace the RoboCasa environment, recorded actions, and MuJoCo model.

For a longer composite task, download one dataset and replay it through the
same Flow:

```bash
python -m robocasa.scripts.download_datasets \
  --tasks PrepareCoffee --split pretrain --source human
python -m examples.advanced.robocasa.app \
  --mode robocasa --task PrepareCoffee --seconds 120 --visualize mjviser
```

Episode 0 currently contains 749 recorded controls and reaches RoboCasa's
success condition near the end of the replay. Scene publication is heavier
than `Lift`; leave the browser open while the kitchen geometry arrives.

Record the same trace without opening a viewer, suitable for remote workers:

```bash
python -m examples.advanced.robocasa.app \
  --mode robocasa --task TurnOnMicrowave --seconds 45 \
  --visualize rerun --rerun-mode record \
  --recording logs/robocasa-replay.rrd
```

Open the artifact later with `rerun logs/robocasa-replay.rrd`.
Record mode uses Retriever's deterministic step recorder; `--seconds * --hz`
is the maximum step budget, and a non-repeating replay stops after its final
recorded action.

Write a compact MP4 for experiment review or a representative documentation
clip:

```bash
pixi run demo-robocasa-video
```

The default output is `logs/robocasa-replay.mp4`. Ordinary recordings remain
ignored under `logs/`; deliberately selected clips can be copied into
`docs-site/public/media/robocasa/` in a separate documentation commit. Adjust
capture cadence with `--image-hz` and playback speed with `--video-fps`.

On macOS, native MuJoCo and offscreen Rerun camera rendering are separate modes
because AppKit cannot safely create both GLFW contexts in one `mjpython`
process. `--viewer --visualize rerun` therefore sends telemetry but not camera
frames to Rerun. MP4 capture also uses the offscreen renderer and cannot be
combined with `--viewer` in the same process.

## What this proves

- Retriever actions enter real RoboCasa physics rather than a toy wrapper.
- Simulator reset, action replay, success checks, images, and telemetry remain
  behind a typed `Flow` boundary.
- End-of-demonstration is an explicit `active=False` signal; this avoids
  accidentally reusing a retained `Latest()` action after the source stops.
- The same graph runs interactively on macOS and headlessly on Linux.
- A policy, planner, or memory system can replace `DemoActionSource` without
  changing the simulator contract.

This is a recorded-demonstration replay, not a learned policy benchmark. Its
purpose is to make the simulator connection visible and reproducible first.

## Related building blocks

- [`robosuite_lift.py`](robosuite_lift.py) provides the smaller closed-loop
  environment-as-Flow and policy-as-Flow prerequisite within this family.
- [`openpi_policy`](../openpi_policy/) defines the existing
  `PolicyObservation -> ActionChunk` policy boundary. A policy-backed RoboCasa
  loop should adapt those chunks into `RoboCasaAction` rather than introduce a
  second policy contract.
- Retriever's `record_session` provides the headless Rerun artifact path; this
  example does not maintain a separate visualization runtime.

## Repository boundary

- **GoldenRetriever** owns these typed Flows, adapters, runnable examples, and
  public demo documentation.
- **external simulator** owns the reproducible Pixi environment, source checkouts, large
  kitchen assets, downloaded datasets, run logs, and machine-specific setup.
- **Retriever Hub** should eventually expose a thin, versioned launcher or
  policy boundary after its inputs, outputs, dependency contract, and smoke
  proof are stable. The simulator stack is intentionally absent from the
  current `golden-retriever` Hub manifest because loading a payload pack must
  not import GUI, simulator, or asset dependencies.

Future TAMP demos should replace `DemoActionSource` with planner and executor
Flows while reusing `RoboCasaSimulator` and either visualization path.
