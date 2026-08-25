# Retriever Embodied Demo Console

This example connects Retriever to an actual RoboCasa simulator through an
offline-first, inspectable execution path:

```text
Goal -> Planner -> Skill plan -> Recorded replay -> Task verification
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
                                                        +-- Retriever web console
                                                        |      +-- mjviser iframe
                                                        +-- native MuJoCo viewer
                                                        +-- Rerun or MP4 capture
```

The ownership boundaries stay narrow: RoboCasa defines kitchen tasks,
demonstrations, and success signals; RoboSuite composes the robot, objects,
arena, and controllers; demonstration mode restores recorded states while
action-replay mode advances MuJoCo; Retriever owns goals, typed
plans, dispatch, replay controls, events, and verification. A standalone
Retriever console owns the experiment UI and embeds mjviser as a replaceable
viewport; mjviser only renders the live `MjModel` and `MjData`. Rerun remains the better view for
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

The launcher highlights atomic tasks such as `CoffeeSetupMug`,
`StartCoffeeMachine`, and `TurnOnMicrowave`, plus composite tasks such as
`PrepareCoffee`, `PackIdenticalLunches`, `LoadDishwasher`,
`OrganizeCondiments`, `StackBowlsCabinet`, and `RestockPantry`. It also
discovers other compatible installed human demonstrations. Missing datasets
remain visible with an unavailable reason instead of failing during launch.
Choose a task and episode, start it, then open the Retriever console on
`http://localhost:8086`.
The console embeds the live mjviser renderer from `http://localhost:8085`.
Starting another scene stops the current viewer first; each RoboCasa task owns
a different environment, demonstration, and MuJoCo model.

The scene launcher is the entry point for switching tasks and frontends. Choose
**Retriever console** for the renderer-independent experiment surface, or
**Viser native** for the dockable local-debug workspace. Both views operate on
the same Retriever controls, plan, events, and simulator process. The embedded
renderer has no ownership over experiment state. At most one simulator child
runs at a time.

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
allow-listed skill plan, the dashboard annotates normalized demonstration
progress with phases such as locate, pick, place, activate, and verify, and
RoboCasa supplies the final success signal. These phase boundaries are curated
replay annotations, not independently detected skill-completion events. The
existing replay command-line interface remains available.

### Execution modes

- **Demonstration replay (available now):** the planner selects a
  validated task manifest, while `DemoActionSource` supplies the recorded
  RoboCasa steps and states. The dashboard advances its annotated timeline from
  replay progress and reports RoboCasa's reward and success signal explicitly.
- **Dynamic skill planning (extension point):** a policy, DMP, TAMP executor,
  or remote planner can replace `DemoActionSource` by producing the existing
  action contract. The console and typed Flow are designed for this mode, but
  the current public example does not claim live low-level planner control.

Selecting `planner="gemini"` changes how the allow-listed `SkillPlan` is
proposed; it does not make the demonstrated trajectory a live model-generated
trajectory.

### Hierarchical composite plans

Composite task manifests are displayed as ordered **subplans**, each containing
one or more dependency-ordered **skills**. For example, `PrepareCoffee` groups
its skills under inspect workspace, position the mug, brew coffee, and verify
outcome. The active subplan and skill advance with the demonstrated trajectory;
completed items remain visible so the dashboard reads as an execution record,
not a static checklist.

The same structure covers curated, genuine RoboCasa composite tasks including
`PackIdenticalLunches`, `LoadDishwasher`, `OrganizeCondiments`,
`StackBowlsCabinet`, and `RestockPantry`. These are task manifests, not invented
dashboard scenarios. Their entries remain visible when datasets are absent,
with the missing dataset or setup requirement shown explicitly; only installed
demonstrations can be executed.

Headless demonstration replay (Linux is the intended remote-worker target):

```bash
python -m examples.advanced.robocasa.app \
  --mode robocasa --task TurnOnMicrowave --seconds 45
```

Environment construction happens before the requested replay budget begins.
The first MJCF and asset initialization can still take roughly 15 seconds on a
laptop.

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

Open `http://localhost:8086` for the complete Retriever console. Its embedded
`http://localhost:8085` viewport renders from the same in-process `MjModel`
and `MjData`; no second simulator is created. Install `mjviser` in the same
external simulation environment as RoboCasa.

The left Retriever console controls the recorded-demonstration Flow. **Run**
pauses, resumes, single-steps, restarts, and changes replay speed. **Plan**
shows ordered subplans and advances their nested skills with the trajectory.
**Graph** renders the current seven-stage pipeline map from `GoalSource`
through `EventSink`, and **Events** shows the compact execution and
verification log.
Restart resets MuJoCo in place and increments the displayed replay cycle. New
browser clients open in the unobstructed agent-facing view; use the **Camera**
control to switch to a tracked third-person robot view or overhead overview.
mjviser's **Track camera** toggle changes between following the robot and a
static world frame.
The Run view also includes a compact planner conversation: submit a text goal,
see which planner produced the allow-listed phases, and receive the terminal
RoboCasa verification result in the same transcript. This conversation is a
view over typed Retriever goals and events, not a second execution system.
Pass `--native-viser-controls` or choose **Viser native** in the launcher to use
the original dockable Viser dashboard. It remains a supported local-debug
frontend; the standalone console is the portable API boundary for other
renderers and remote machines.

Use `demo-robocasa-scenes` to switch between curated atomic and composite tasks
or another installed dataset. `CoffeeSetupMug` is the initial atomic
pick-place example; `PrepareCoffee` is the installed composite reference run.
The viewer stays bound to one task because switching tasks replaces the
RoboCasa environment, recorded actions, and MuJoCo model.

For a longer composite task, download one dataset and replay it through the
same Flow:

```bash
python -m robocasa.scripts.download_datasets \
  --tasks PrepareCoffee --split pretrain --source human
python -m examples.advanced.robocasa.app \
  --mode robocasa --task PrepareCoffee --seconds 120 --visualize mjviser
```

The selected installed episode reports its recorded step count and RoboCasa
success signal in the console. Scene publication is heavier than `Lift`; leave
the browser open while the kitchen geometry arrives.

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

- Retriever deterministically restores recorded RoboCasa simulator states for
  the public demonstration path. The typed action contract remains available
  for a future closed-loop executor.
- Simulator reset, recorded-state replay, optional action replay, success
  checks, images, and telemetry remain behind a typed `Flow` boundary.
- End-of-demonstration is an explicit `active=False` signal; this avoids
  accidentally reusing a retained `Latest()` action after the source stops.
- The same graph is tested interactively on macOS and is designed for
  headless Linux workers.
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
- **Linux:** intended target for interactive or headless replay; verify the
  installed graphics stack and datasets on each worker.
- **Windows:** WSL2 is experimental and currently unverified; native Windows is
  not a tested target.

The console has no ROS or Isaac Sim dependency. The referenced Isaac Sim
Gemini Robotics project informed interaction patterns only; this implementation
uses Retriever, RoboCasa, MuJoCo, and mjviser directly.

The renderer boundary is intentionally replaceable. The console speaks only to
Retriever's JSON control and state API, while the simulator supplies a viewport
URL. This keeps future WebGPU, streamed-video, Rerun, or remote-renderer adapters
from leaking into planning and experiment controls.

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
- **External simulation environments** own reproducible dependency locks,
  source checkouts, large
  kitchen assets, downloaded datasets, run logs, and machine-specific setup.
- **Retriever Hub** should eventually expose a thin, versioned launcher or
  policy boundary after its inputs, outputs, dependency contract, and smoke
  proof are stable. The simulator stack is intentionally absent from the
  current `golden-retriever` Hub manifest because loading a payload pack must
  not import GUI, simulator, or asset dependencies.

Future TAMP demos should replace `DemoActionSource` with planner and executor
Flows while reusing `RoboCasaSimulator` and either visualization path.

