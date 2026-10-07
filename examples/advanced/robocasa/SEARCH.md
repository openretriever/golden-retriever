# RoboCasa multi-drawer search (D1)

Can a robot search several possible containers, remember what it has already
looked in, and stop when it finds the target — reproducibly, and inspectably?

This lane is being built in phases. **Phases 0–4 are complete**: the search
program runs end to end both against a mock world and against a real RoboCasa
kitchen, deterministically, with all five catalog metrics derived from traces.
Only the review packaging (video, async update) remains.

## Status

| Phase | What it covers | State |
| --- | --- | --- |
| 0 | Environment stood up, one real task replayed, RoboCasa feasibility settled | done |
| 1 | Belief, policy, trace — mock-first, no simulator | done |
| 2 | Three fixed scenes, seeded target placement | done |
| 3 | Bind to real physics | done |
| 4 | Metrics and determinism | done |
| 5 | Video and review packaging | not started |

## Quick start

No simulator, assets, GPU or network required:

```bash
pixi run demo-robocasa-search-mock
```

```text
scene=three_drawer_stack seed=3 policy=nearest-first target='pepper_shaker'

  start           belief drawer_top=0.25 drawer_middle=0.25 drawer_bottom=0.25 cabinet_left=0.25
  plan            drawer_bottom -> drawer_middle -> drawer_top -> cabinet_left
  step 1  open drawer_bottom    -> empty  (+0.69m, 25%)
          why: nearest candidate, cost=0.69
          belief drawer_top=0.33 drawer_middle=0.33 drawer_bottom=0.00 cabinet_left=0.33
  step 2  open drawer_middle    -> empty  (+0.25m, 50%)
          why: nearest candidate, cost=0.25
          belief drawer_top=0.50 drawer_middle=0.00 drawer_bottom=0.00 cabinet_left=0.50
  step 3  open drawer_top       -> empty  (+0.25m, 75%)
          why: nearest candidate, cost=0.25
          belief drawer_top=0.00 drawer_middle=0.00 drawer_bottom=0.00 cabinet_left=1.00
  step 4  open cabinet_left     -> FOUND  (+0.84m, 100%)
          why: nearest candidate, cost=0.84
          belief drawer_top=0.00 drawer_middle=0.00 drawer_bottom=0.00 cabinet_left=1.00
  stop            target found  [SUCCESS]

  status=SUCCESS success=True containers_opened=4 repeated=0 steps=4 travel=2.03m progress=100% replans=0
```

Every ruled-out container's mass redistributes over the rest, so the belief is
auditable line by line. Try `--scene galley_run`, `--policy fixed-order`, a
different `--seed`, or `--trace logs/search.jsonl` for the machine-readable
trace.

## 1. Verified setup

Every command below was run end to end on macOS arm64 (M-series, macOS 14.4)
against `main` at `b7356af`. Run them in order from the repository root.

```bash
pixi install --locked -e robocasa
pixi run --locked -e robocasa robocasa-smoke
pixi run --locked -e robocasa robosuite-setup-macros
pixi run --locked -e robocasa robocasa-setup-macros
pixi run --locked -e robocasa robocasa-download-assets          # see blocker 1
pixi run --locked -e robocasa robocasa-download-turn-on-microwave
```

`robocasa-smoke` is the preflight and should print:

```text
[ok] mujoco 3.3.1
[ok] robosuite 1.5.2
[ok] robocasa 1.0.1
[ok] mjviser 0.0.14
[ok] fastapi 0.141.1
[ok] uvicorn 0.52.4
[ok] headless RoboSuite Lift step (action_dim=7)
[ok] mjviser bridge start/update/stop
[ok] RoboCasa asset downloader
```

Assets land in `.pixi/envs/robocasa/.../robocasa/models/assets` (**10 GB** on
disk once extracted) and demonstrations in `.../site-packages/datasets/v1.0/`.

## 2. One real task replayed

```bash
pixi run --locked -e robocasa demo-robocasa-replay
```

Replays the recorded `TurnOnMicrowave` human demonstration through real
RoboCasa physics. It reaches RoboCasa's own success condition partway through
and holds it to the end:

```text
[robocasa step=0088] progress=81.5% reward=0.000 success=False
[robocasa step=0092] progress=85.2% reward=0.000 success=False
[robocasa step=0093] progress=86.1% reward=1.000 success=True
[robocasa step=0108] progress=100.0% reward=1.000 success=True
Closed the Retriever-connected RoboCasa simulator.
```

This satisfies the D1 milestone bullet "run one existing RoboCasa task and
record the exact setup command."

## 3. Feasibility: can RoboCasa support multi-drawer search?

Three questions had to be answered before committing to an approach. All three
are **yes**, probed directly against the installed simulator.

### Q1. Can containers be enumerated by name? — yes

`env.fixtures` is a `dict` of named fixtures. One kitchen
(`PickPlaceCounterToDrawer`) exposed **101 fixtures, 42 of them containers**:

```text
stack_04_main_group_1_1   Drawer
stack_04_main_group_1_2   Drawer
stack_04_main_group_1_3   Drawer
hingecabinet_5_main_group_1   HingeCabinet
singlecabinet_2_main_group_1  SingleCabinet
...
```

A single kitchen contains **11 independently addressable drawers**, already
grouped into three-drawer stacks. D1 needs no bespoke scene — the multi-drawer
geometry ships with RoboCasa.

### Q2. Can open/closed state be read back and driven? — yes

`Drawer` and `HingeCabinet` expose the full state API:

```python
set_door_state(self, min, max, env)
get_door_state(self, env)                          # -> {joint: normalized qpos}
is_open(self, env, joint_names=None, th=0.90)
is_closed(self, env, joint_names=None, th=0.005)
open_door(self, env, min=0.9, max=1, partial_open=True)
close_door(self, env, min=0.0, max=0.0)
```

Verified driving one drawer through its range:

```text
target drawer: stack_2_main_group_2
  initial:           state={'door': 0.909}  open=False closed=False
  after close_door:  state={'door': -0.0}   open=False closed=True
  after open_door:   state={'door': 0.868}  open=False closed=False
  after set(0,0):    state={'door': 0.0}    open=False closed=True
```

> **Gotcha.** `open_door()` defaults to `partial_open=True` and settles at
> ~0.868, but `is_open()` tests against `th=0.90` — so **a drawer opened with
> the default call does not report as open**. Use
> `open_door(env, min=1.0, max=1.0, partial_open=False)`, or compare
> `get_door_state()` against your own threshold. D1's "is this container open"
> check must not rely on `is_open()` naively.

Note also that `SingleCabinet` and `OpenCabinet` have **no** `get_door_state`;
they expose `is_open` / `is_closed` / `get_joint_state` instead. Any container
abstraction has to handle that per-class variation.

### Q3. Can an object be hidden inside a *named* container? — yes

This is the crux: D1 needs a target placed in a seed-chosen drawer.
`PickPlaceDrawerToCounter` already does exactly this, and its object config
points `placement["fixture"]` straight at a named `Drawer`:

```text
object 'obj'   groups=('tool', 'utensil')
  placement['fixture'] = Drawer  name='stack_08_main_group_1_2'

object 'distr'
  placement['fixture'] = Counter name='counter_3_main_group_1'
  placement['sample_region_kwargs']['ref'] = Drawer name='stack_08_main_group_1_2'
```

So `placement["fixture"] = <the chosen Drawer>` is the seam for hiding the
target, and `sample_region_kwargs["ref"]` positions distractors relative to it.

**Consequence: D1 should use RoboCasa's own kitchens rather than a bespoke
scene.** The containers, their state API, and in-container placement all exist
upstream.

## 4. Phase 1: the search program

Built mock-first, so all of it is exercised by `pixi run test` in CI, which
never installs the simulator.

| File | What it holds |
| --- | --- |
| `search_scenes.py` | containers and scenes as plain data — no runtime import at all, so tests and docs share one definition. `target_container(seed)` is a SHA-256 of `(scene_id, seed)`, not `hash()` or `random`, so placement is identical across processes and platforms |
| `search_belief.py` | `SearchBelief`, a discrete distribution over containers. Looking and not finding drives a container's mass to zero and renormalizes; finding collapses it. Frozen — updates return a new belief |
| `search_policy.py` | belief in, next container out. `fixed-order` (baseline) and `nearest-first` (cost-ordered), sharing one stopping rule so any metric gap is attributable to ordering alone |
| `search_world.py` | the `SearchWorld` boundary plus `MockSearchWorld`. Mirrors `method_harness.EnvironmentAdapter`'s `reset`/`verify`/`close`, with `inspect(container_id)` in place of `step(action)`, plus `status()` for progress and failure |
| `search_trace.py` | append-only JSONL plus a renderer. Every inspection records the belief *before and after*, so the update is auditable rather than asserted |
| `search_plan.py` | `SearchPlan` plus `plan_ahead`, which rolls the policy forward over a hypothetical belief to get its intended order without touching the world |
| `search_app.py` | the episode loop and CLI |

### The plan, and why it is not a `SkillPlan`

A search plan cannot be a fixed script: its length is unknown until the target
turns up. `embodied.SkillPlan.validate()` requires contiguous progress
fractions covering `[0, 1]`, which assumes the step count is known up front —
so satisfying it would mean fabricating fractions for steps that may never run,
or weakening a contract other lanes depend on. `SearchPlan` is therefore an
open-ended payload; `SkillPlan` stays the right shape for *displaying* a
committed plan, and projecting onto it is a display concern.

The plan is a prediction, and the trace records both it and the route actually
taken. Each time the remaining order stops matching the prediction, that is a
**replan** — the catalog's demo criteria ask for at least one system metric
alongside task success and name replans first.

**Replans are 0 for both shipped policies, and that is the correct answer.**
`plan_ahead` assumes each container comes back empty, which is exactly what
happens until the target appears, so with noiseless observations and policies
that depend only on belief and position the lookahead is exact *by
construction*. The counter becomes informative when observations can be wrong
(D3's false-negative condition) or a policy carries internal state. Since an
instrument that only ever reads zero is indistinguishable from a broken one,
`test_robocasa_search_plan.py` drives a deliberately stateful policy to prove
the counter fires.

## 5. Phase 2: three fixed scenes

Each scene measures something different, and container declaration order is
never distance order — if a scene listed containers nearest-first, the two
policies would collapse into one and the comparison would measure nothing.

| Scene | Containers | Why it exists |
| --- | --- | --- |
| `three_drawer_stack` | 4 | A drawer stack plus a cabinet; modest cost spread |
| `galley_run` | 6 | Spread down a counter, ~3x cost from end to end — where cost ordering should pay off most |
| `island_ring` | 4 | Every container exactly equidistant. The **control**: with nothing to order by, the policies must agree |

Target placement is a pure function of `(scene_id, seed)`, and every container
is reachable across seeds — asserted per scene.

### What the policies actually differ on

This is worth stating plainly, because the obvious metric is the wrong one.

**Container count cannot be improved by ordering.** With a uniformly placed
target and a belief that never re-opens, the expected number of containers
opened is `(n+1)/2` *whatever order you visit in*. Measured over 200 seeds per
cell, that is exactly what happens — and an early 12-seed run that appeared to
show a 2.33-vs-3.00 advantage was small-sample noise.

What ordering moves is **distance travelled**:

| Scene | Policy | Containers opened | Travel |
| --- | --- | --- | --- |
| `three_drawer_stack` | fixed-order | 2.48 | 1.53 m |
| `three_drawer_stack` | nearest-first | 2.42 | **1.18 m** |
| `galley_run` | fixed-order | 3.37 | 5.92 m |
| `galley_run` | nearest-first | 3.68 | **2.98 m** |
| `island_ring` | fixed-order | 2.46 | 3.28 m |
| `island_ring` | nearest-first | 2.46 | 3.28 m |

200 seeds per cell, every episode successful. Container count is flat; travel
separates by 23% and 50%. On the equidistant control the two policies are
indistinguishable, which is what shows the gap is geometry and not an artifact
of the harness.

Both facts are locked into tests, so a future policy cannot quietly claim
credit on the order-invariant metric.

### The six interfaces the milestone asks for

| Interface | Where |
| --- | --- |
| observation | `SearchWorld.observe()` → `SearchObservation` (scene, containers opened, holding target, step) |
| action | `SearchPolicy.decide()` → `SearchDecision` (`inspect` a named container, or `stop`) |
| reset | `SearchWorld.reset(scene, seed)` |
| success | `SearchWorld.verify()` → `SearchVerification`, decided by the world, not the belief |
| progress | `SearchWorld.status()` → `ExecutionStatus.progress`, 0.0–1.0 |
| failure | `SearchWorld.status()` → `status="FAILURE"` plus `error_message` |

Progress and failure use `retriever_typing.ExecutionStatus` — the repo's
exported standard type for this, already carrying `status` / `progress` /
`error_message` — rather than a local enum, because `AGENTS.md` makes type
identity the contract.

`progress` is the fraction of the container space resolved, so it rises as
containers are ruled out and reads 1.0 the moment the search is decided, by
either outcome. It is derived from counts alone, which keeps it deterministic
and therefore safe to write into a trace (unlike wall-clock time, which is
deliberately kept out — see Phase 4).

Two distinct failure modes are recorded rather than raised:

- **exhaustion** — every container inspected and the target was never there
- **world error** — the backend raised mid-inspection; the trace keeps the
  decision that led to it and terminates `FAILURE`

Neither is reachable on the happy path, since placement always puts the target
somewhere and the mock never breaks, so both are tested with deliberately
broken worlds in `test_robocasa_search_status.py`.

### Why the policy acts on containers, not action rows


`MethodAdapter.predict()` returns an `ActionChunk` — raw action rows bounded by
`SafetyEnvelope`. A search policy decides "open `drawer_middle`", which is not
an action chunk. Rather than force the search logic down to joint level, the
policy sits *above* the harness and a scripted skill executor (Phase 3) expands
each chosen container into chunks. The harness still runs the trial and still
produces `TrialReport` / `HarnessEvent`.

This is open question 2 below and is worth confirming before Phase 3.

## 6. Phase 3: real RoboCasa physics

Same `SearchWorld` boundary, so the belief, the policies and the trace run
unchanged. RoboCasa is imported lazily inside `search_robocasa_world.py`, so the lane
stays import-safe and CI stays simulator-free.

```bash
pixi run --locked -e robocasa python -m examples.advanced.robocasa.search_app \
  --backend robocasa --layout 5 --containers 5 --seed 2
```

The scene is **derived from the live kitchen** — container ids, positions and
count are real fixtures, so travel costs are real distances. Reproducibility
comes from pinning `layout_ids`, `style_ids` and RoboCasa's own `seed`; where
the target hides stays a pure function of `(scene_id, seed)` in `search_scenes.py`, so
the mock and real backends agree on the answer (asserted in the sim tests).

Opening a container drives its joint through RoboCasa's fixture API rather than
grasping the handle. D1 measures search efficiency; the catalog assigns
contact-level manipulation to D4, and `examples/advanced/robocasa_drawer/` is
the reference for what real contact-based opening costs.

### Results on layout 5, five containers, 48 seeds

| Policy | Success | Containers opened | Travel |
| --- | --- | --- | --- |
| fixed-order | 48/48 | 3.23 | 6.57 m |
| nearest-first | 48/48 | 2.75 | **4.33 m** |

Container count straddles the theoretical `(n+1)/2 = 3.0` for both, as it must.
Travel falls 34%. Same seed replays a byte-identical trace, and the world's own
geometry — not the agent's belief — agrees with the outcome on all 48 seeds.

### Three traps this hit, all of which silently produce meaningless numbers

- **`robot0_base` is a parked dummy.** On a mobile-base robot it sits at
  `(10, 10, 0)` while the arm really hangs off `mobilebase0_base`. Reading the
  wrong body put the robot 14 m from its own kitchen and made the first leg of
  every search cost 14 m.
- **Drawers come in stacks.** Members of a stack share x and y and differ only
  in height, so "the nearest N drawers" is usually one stack, every travel cost
  lands within a few centimetres, and the policy comparison collapses. The
  world now takes one drawer per stack. Layout 1 has only two stacks and cannot
  spread four containers at all; layout 5 has five.
- **Selecting by distance and then emitting in that order** makes the scene's
  declaration order *be* distance order, which silently turns `fixed-order`
  into `nearest-first`. Containers are now emitted in name order. Both the mock
  and sim suites assert declaration order is not distance order.

### Tests

```bash
pixi run --locked -e robocasa test-robocasa-search-sim   # 16 tests, ~25 s
```

Opt-in behind `RUN_ROBOCASA_SEARCH_TESTS=1`, matching
`test_robocasa_replay_data.py`. CI never installs the simulator, so
`test_robocasa_search.py` covers the same logic mock-first and is what gates.

## 7. Phase 4: metrics and determinism

```bash
pixi run demo-robocasa-search-metrics        # 1,200 mock episodes + determinism
```

`search_metrics.py` computes every catalog metric **from a trace**, never from
instrumented internals, and `EpisodeMetrics.from_jsonl` recomputes them from a
written trace file. A test asserts the two agree — which is the actual proof
behind the done-definition's "inspectable without reading internal simulator
code".

One deliberate omission: **elapsed wall-clock time is not written into the
trace.** It varies run to run and would break the byte-identical replay check
that the determinism test depends on, so timing is measured around the episode
instead. A test pins that too.

### Results — mock, 3 scenes × 2 policies × 200 seeds

```text
scene                 policy            n  success  opened  repeat  unnec   travel      ms
------------------------------------------------------------------------------------------
galley_run            fixed-order     200    100%    3.37    0.00   0.16     5.92     0.0
galley_run            nearest-first   200    100%    3.68    0.00   0.21     2.98     0.1
island_ring           fixed-order     200    100%    2.46    0.00   0.21     3.28     0.0
island_ring           nearest-first   200    100%    2.46    0.00   0.21     3.28     0.0
three_drawer_stack    fixed-order     200    100%    2.48    0.00   0.23     1.53     0.0
three_drawer_stack    nearest-first   200    100%    2.42    0.00   0.23     1.18     0.0
```

### Results — real RoboCasa kitchen, 48 seeds

The same metrics module, unchanged, against real physics:

```text
scene                 policy            n  success  opened  repeat  unnec   travel      ms
------------------------------------------------------------------------------------------
robocasa_layout5      fixed-order      48    100%    3.23    0.00   0.23     6.57  2223.9
robocasa_layout5      nearest-first    48    100%    2.75    0.00   0.15     4.33  1929.8
```

Determinism passes on every scene and policy in both backends, including three
consecutive runs under real physics.

### How to read these columns

- **success** — 100% everywhere. The task is always solvable; success alone
  does not discriminate between policies and is a floor, not a result.
- **opened** — flat within noise, by construction. With a uniformly placed
  target the expectation is `(n+1)/2` regardless of visit order, so this column
  measures whether the *belief* works (never re-opening), not the policy.
- **repeat** — 0.00 everywhere. The belief rules out what it has seen, so a
  container is never opened twice.
- **unnec** — inspections whose outcome the belief already implied. Roughly
  equal across policies at ~0.2 because it reflects the **stopping rule**, not
  ordering: termination requires physically finding the target, so when the
  target sits in the last-searched container the agent must open it even though
  elimination already gave the answer. An agent that inferred and stopped would
  drive this to zero — a real comparison for D3 or D6.
- **travel** — the column ordering actually moves: 23% better on
  `three_drawer_stack`, 50% on `galley_run`, 34% on the real kitchen, and
  exactly zero on the equidistant control.

## Known blockers

1. **The asset downloader is unreliable and silently incomplete.**
   `robocasa-download-assets` fetches six archives in order — `tex`,
   `tex_generative`, `fixtures_lw`, `objs_objaverse`, `objs_aigen`, `objs_lw`.
   It uses `urllib.urlretrieve`, which has **no resume**: when the Box CDN drops
   a connection the archive restarts from zero. `objs_aigen` (5.8 GB) stalled
   permanently at 70%, which meant `objs_lw` — the last in the list — never
   downloaded at all. Kitchens then fail at reset with
   `FileNotFoundError: .../objects/lightwheel/stool/Stool018/model.xml`.

   Fetch a single archive with
   `python -m robocasa.scripts.download_kitchen_assets --type objs_lw`. If that
   also restart-loops (it retried 17 times here, never passing 73%), resolve the
   URL from `DOWNLOAD_ASSET_REGISTRY` and pull it with `curl -L -C -`, which
   resumes; Box does serve HTTP 206 range requests. Then unzip into
   `models/assets/objects/` and delete the archive.

   `objs_aigen` is still missing here and nothing so far has needed it.

2. **Kitchen layout is randomized per reset.** The same task gave the target
   drawer `stack_08_main_group_1_2` on one reset and `stack_2_main_group_2` on
   another. There is no seeding anywhere in the Retriever RoboCasa lane, so
   fixed scenes require seed plumbing that does not exist yet.

3. **The existing lane replays; it cannot search.** `RoboCasaSimulator`
   reconstructs the environment verbatim from a recording's `env_kwargs` and
   teleports to recorded states rather than calling `env.step()`. This lane
   therefore builds its own kitchen rather than extending that one. Wiring the
   search into `EXECUTION_MODES`' declared-but-unimplemented `live_planning`
   mode is still open, and depends on the open question below about whether a
   container-level policy belongs above `MethodAdapter` or inside it.

5. **Opening is state-driven, not contact-driven.** Containers open via
   `set_door_state`, so the search is honest about *which* containers get
   opened and in what order, but not about whether a real arm could open them.
   The target also does not ride the drawer out when it slides. Fine for D1's
   metrics; D4 owns closing that gap.

4. **`objs_aigen` incomplete**, as above — untested whether any kitchen needs it.

## Open questions

Three questions for the project lead, each of which changes roughly a week of
work:

1. **Substrate.** Q1–Q3 say RoboCasa's own kitchens can host D1 directly. Is
   implementing `live_planning` against them the intended path, rather than
   building on the separate `examples/robocasa-drawer-v2` scene?
2. **Interface.** Should D1's environment interface conform to
   `method_harness.EnvironmentAdapter` (`reset`/`step`/`verify`/`close`)? It is
   not on `main` yet, and a search policy decides over *containers*, not
   `ActionChunk` rows.
3. **Skill fidelity.** Is a scripted `open`/`inspect`/`close` layer acceptable
   for D1, leaving real low-level skill execution to D4?
