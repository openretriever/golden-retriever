# Retriever Embodied Demo Console

This example connects Retriever to an actual RoboCasa simulator through an
offline-first, inspectable execution path:

```text
Goal -> Planner -> Skill plan -> Verified RoboCasa replay
```

The default planner maps a supported goal to a validated task manifest without
network access or credentials. Retriever then dispatches a known-good RoboCasa
demonstration and reports progress, step results, reward, and task verification
in the browser console.

```text
GoalSource -> EmbodiedPlanner -> SkillDispatcher -> DemoActionSource
                                                        |
                                                      Latest
                                                        v
                         EventSink <- TaskVerifier <- RoboCasaSimulator
                                                        |
                                                        +-- mjviser browser scene
                                                        +-- native MuJoCo viewer
                                                        +-- Rerun or MP4 capture
```

The ownership boundaries stay narrow: RoboCasa defines kitchen tasks,
demonstrations, and success signals; RoboSuite composes the robot, objects,
arena, and controllers; MuJoCo advances physics; Retriever owns goals, typed
plans, dispatch, replay controls, events, and verification; mjviser renders the
live `MjModel` and `MjData` in the browser. Rerun remains the better view for
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

## 4. Choose a scene in the browser

Start the local scene launcher, then open `http://localhost:8084`:

```bash
retriever run demo-robocasa-scenes
```

The launcher highlights the curated `PrepareCoffee`, `CoffeeSetupMug`,
`StartCoffeeMachine`, and `TurnOnMicrowave` tasks, and also discovers other
compatible installed human demonstrations. Missing datasets remain visible
with an unavailable reason instead of failing during launch. Choose a task and
episode, start it, then open the live viewer on `http://localhost:8085`.
Starting another scene stops the current viewer first; each RoboCasa task owns
a different environment, demonstration, and MuJoCo model.

The scene launcher is the entry point for switching tasks. The viewer remains
the focused experiment console for replay controls, camera presets, and the
live Retriever graph. At most one simulator child runs at a time.

## 5. Run a real replay directly

The concise Python entry point uses the offline planner by default:

```python
from examples.advanced.robocasa.app import run

run(
    task="PrepareCoffee",
    episode=0,
    planner="offline",
    visualize="mjviser",
    open_browser=True,
)
```

The replay remains deterministic and inspectable: the planner selects an
allow-listed skill plan, the dispatcher advances phase markers such as locate,
pick, place, activate, and verify, and RoboCasa supplies the final success
signal. The existing replay command-line interface remains available.

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

The left Retriever console controls the actual recorded-action Flow. **Run**
pauses, resumes, single-steps, restarts, and changes replay speed. **Plan**
shows the ordered skill phases and their results. **Graph** renders the live
seven-stage typed Flow from `GoalSource` through `EventSink`, and **Events**
shows the compact execution and verification log. Restart resets MuJoCo in
place and increments the displayed replay cycle. New browser clients open in
the unobstructed agent-facing view; use the **Camera** control to switch to a
tracked third-person robot view or overhead overview. mjviser's **Track
camera** toggle changes between following the robot and a static world frame.

Use `demo-robocasa-scenes` to switch between `PrepareCoffee`,
`CoffeeSetupMug`, `StartCoffeeMachine`, `TurnOnMicrowave`, or another installed
dataset. `CoffeeSetupMug` is the initial atomic pick-place example;
`PrepareCoffee` is the composite example. The viewer stays bound to one task
because switching tasks replaces the RoboCasa environment, recorded actions,
and MuJoCo model.

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

## Optional Gemini planner

Set `planner="gemini"` to use the optional Gemini embodied-reasoning planner.
Its output is parsed as a structured `SkillPlan` and rejected unless every
step uses the same allow-listed skills accepted by the offline planner. It
cannot execute code or send arbitrary MuJoCo controls. If credentials or the
optional client are unavailable, the console falls back to the offline plan.

## Platform support

- **macOS:** primary local path for interactive mjviser and native MuJoCo
  inspection.
- **Linux:** interactive or headless replay, including remote experiment
  workers and saved Rerun artifacts.
- **Windows:** supported through WSL2 where the Linux simulator and browser
  networking requirements are available; native Windows is not a tested
  target.

The console has no ROS or Isaac Sim dependency. The referenced Isaac Sim
Gemini Robotics project informed interaction patterns only; this implementation
uses Retriever, RoboCasa, MuJoCo, and mjviser directly.

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

