# Retriever Embodied Demo Console

This example connects Retriever to the RoboCasa simulator through an
offline-first, inspectable execution path:

```text
Goal -> Planner -> Skill plan -> Recorded replay -> Task verification
```

RoboCasa defines the kitchen tasks, demonstrations, and success signals.
RoboSuite and MuJoCo run the simulation. Retriever owns the typed plan,
execution controls, events, verification, and browser console. mjviser renders
the same live `MjModel` and `MjData`; it does not start a second simulator.

## Quick start: no simulator required

From the repository root, run the deterministic contract checks:

```bash
pixi run demo-robosuite-mock
pixi run demo-robocasa-mock
```

The default Pixi environment intentionally contains no simulator or kitchen
assets. This keeps normal tests and the mock examples lightweight.

## Install the simulator environment

Use the dedicated Pixi environment for every real simulator command. Do not
enter a Pixi shell or manually install packages with `pip`:

```bash
pixi install --locked -e robocasa
pixi run --locked -e robocasa robocasa-smoke
```

The `robocasa` environment installs pinned revisions of
[RoboSuite](https://github.com/ARISE-Initiative/robosuite) and
[RoboCasa](https://github.com/robocasa/robocasa), plus MuJoCo, mjviser, and the
console dependencies. `robocasa-smoke` imports that stack and constructs a
headless RoboSuite `Lift` environment without downloading kitchen assets.

The lock covers macOS arm64 and Linux x86-64. Native Windows is not tested;
WSL2 may work but is not part of the verified path.

## Download RoboCasa assets and data

The lock covers Python and native dependencies. RoboCasa's kitchen assets and
human demonstrations are separate upstream downloads and are not pinned by
this repository. Run their setup commands through the same environment:

```bash
pixi run --locked -e robocasa robosuite-setup-macros
pixi run --locked -e robocasa robocasa-setup-macros
pixi run --locked -e robocasa robocasa-download-assets
pixi run --locked -e robocasa robocasa-download-turn-on-microwave
pixi run --locked -e robocasa robocasa-download-coffee-setup-mug
pixi run --locked -e robocasa robocasa-download-prepare-coffee
```

The kitchen assets require roughly 10 GB. The final three commands download
the demonstrations used by the direct replay and real-data checks.
See the official [RoboCasa installation guide](https://robocasa.ai/docs/introduction/installation.html)
and [dataset guide](https://robocasa.ai/docs/build/html/datasets/using_datasets.html)
for upstream details.

## Run the browser console

Launch the task catalog and open `http://localhost:8084`:

```bash
pixi run --locked -e robocasa demo-robocasa-scenes
```

The launcher shows curated atomic tasks such as `CoffeeSetupMug`,
`StartCoffeeMachine`, and `TurnOnMicrowave`, and composite tasks such as
`PrepareCoffee`, `PackIdenticalLunches`, `LoadDishwasher`,
`OrganizeCondiments`, `StackBowlsCabinet`, and `RestockPantry`. Installed
demonstrations are runnable; unavailable datasets remain visible with a clear
reason instead of failing at launch.

Choose **Retriever console** for the portable dashboard or **Viser native** for
the dockable local workspace. Both operate on the same Retriever state and one
simulator process. Switching tasks stops the previous process before starting
the next one.

The console provides four views:

- **Run** controls pause, resume, single-step, restart, playback speed, camera,
  and text goals.
- **Plan** shows ordered subplans and skills advancing with replay progress.
- **Graph** shows the current Retriever Flow.
- **Events** shows dispatch, completion, failure, and verification events.

## Direct commands

Run the small real RoboSuite task first:

```bash
pixi run --locked -e robocasa demo-robosuite-lift
```

Run a headless RoboCasa replay:

```bash
pixi run --locked -e robocasa demo-robocasa-replay
```

Run the same replay in the browser, Rerun, or an MP4 recorder:

```bash
pixi run --locked -e robocasa demo-robocasa-web
pixi run --locked -e robocasa demo-robocasa-rerun
pixi run --locked -e robocasa demo-robocasa-video
```

Run the installed composite example:

```bash
pixi run --locked -e robocasa demo-robocasa-composite-web
```

After downloading `CoffeeSetupMug` and `PrepareCoffee`, check episode 0 of an
atomic and a composite task against RoboCasa's success signal:

```bash
pixi run --locked -e robocasa test-robocasa-replay-data
```

This opt-in check restores every recorded state and requires the final state of
both demonstrations to satisfy RoboCasa's task verifier. CI omits the large
asset and dataset downloads, so it does not run this task automatically.

The default MP4 path is `logs/robocasa-replay.mp4`. Routine recordings remain
ignored under `logs/`.

### Bounded methods harness

The scene catalog also exposes a small methods harness for comparing embodied
execution approaches without coupling the dashboard to one simulator or model.
`MethodHarness` accepts the existing `PolicyObservation -> ActionChunk`
boundary, validates horizon, shape, finite values, magnitude, and source, then
records ordered events and native task verification. The included scripted
privileged cube lift is a deterministic integration smoke, not a learned-policy
result. Generated Python and arbitrary simulator commands are not accepted.

See [`REPRODUCED_RESULTS.md`](REPRODUCED_RESULTS.md) for the tested environment,
exact command, and current result.

## Native MuJoCo viewer

The browser and headless commands above use ordinary Python on macOS and
Linux. The inspection task selects `mjpython` for MuJoCo's passive viewer on
macOS and ordinary Python on Linux:

```bash
pixi run --locked -e robocasa robocasa-inspect
```

The inspection module explicitly imports `mujoco.viewer`; importing `mujoco`
alone does not load that submodule. The script also closes the RoboSuite
environment when the viewer exits.

Native viewer and offscreen camera capture remain separate on macOS because
they create incompatible AppKit/GLFW contexts in one process.

## Python API

The concise entry point uses the offline planner by default:

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

The offline planner maps a supported goal to an allow-listed task manifest
without network access or credentials. It dispatches a recorded RoboCasa
demonstration and annotates progress with phases such as locate, pick, place,
activate, and verify. These are curated replay annotations, not independently
detected skill completions.

The simulator environment includes the optional Gemini client. Set
`GEMINI_API_KEY` and use `planner="gemini"` to propose an allow-listed
`SkillPlan`. Invalid tool calls are rejected; unavailable credentials or
operational failures return an `offline`-sourced plan, which the console labels
accordingly. The model cannot execute code or send arbitrary MuJoCo controls.

## Execution model

```text
GoalSource -> EmbodiedPlanner -> SkillDispatcher -> DemoActionSource
                                                        |
                                                      Latest
                                                        v
                         EventSink <- TaskVerifier <- RoboCasaSimulator
                                                        |
                                                        +-- Retriever console
                                                        +-- mjviser
                                                        +-- Rerun or MP4
```

The current public path is deterministic demonstration replay. The planner
selects a validated manifest, `DemoActionSource` restores recorded states, and
RoboCasa supplies reward and success. A policy or motion executor can later
replace `DemoActionSource` through the existing typed action boundary without
changing the console or simulator contract.

Composite manifests group dependency-ordered skills into subplans. The active
subplan advances with the selected demonstration; completed items remain
visible as an execution record.

## What this proves

- The checked-in lock creates one reproducible simulator dependency environment.
- Mock contracts work without RoboCasa, assets, cameras, or a GUI.
- Real replay, reset, recorded states, images, and success checks remain behind
  typed Retriever Flow boundaries.
- One simulator owns physics while the console and visualization adapters
  consume its state.
- Missing datasets and disconnected viewers are explicit UI states.

The cross-platform smoke check constructs RoboSuite `Lift`, starts and stops
mjviser, and exercises the launcher API. It does not download the roughly 10 GB
kitchen assets or replay a RoboCasa demonstration in CI.

This is a recorded-demonstration replay, not a learned-policy benchmark.

## Related examples

- [`robosuite_lift.py`](robosuite_lift.py) is the smaller closed-loop
  environment-as-Flow and policy-as-Flow prerequisite.
- [`openpi_policy`](../openpi_policy/) defines the existing
  `PolicyObservation -> ActionChunk` policy boundary.
- Retriever's `record_session` provides the headless Rerun artifact path.

GoldenRetriever contains the typed Flows, adapters, runnable examples, and
public documentation. Large assets, downloaded datasets, recordings, and
machine-specific caches remain outside version control.
