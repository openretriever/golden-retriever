# Running the study on your own account

Everything here runs on a laptop except the VLA arm. A model-as-policy sweep is
CPU-bound simulation plus API calls, so the only thing it needs from a cloud
account is a Vertex AI project — no VM, no GPU. Measured cost is **$0.19 per
episode** over 28 episodes and eleven models.

## Setup

```bash
git clone https://github.com/openretriever/golden-retriever.git
cd golden-retriever
git checkout feature/robocasa-pnp-policy-study
pixi install -e robocasa          # ~20 min; pulls robosuite, robocasa, torch
pixi run -e robocasa robocasa-download-assets   # ~23 GB of kitchen assets, once
```

Then point it at your own Google Cloud project. Two separate credentials: the
CLI uses one and the Python client uses the other, and fixing only the first
leaves the second prompting.

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable aiplatform.googleapis.com
```

Verify before spending anything:

```bash
pixi run -e robocasa robocasa-smoke          # simulator renders, assets present
pixi run -e robocasa test-robocasa-study     # 310 tests, no network, no keys
pixi run -e robocasa study-pnp-preflight     # 13 checks against a live model
```

`study-pnp-preflight` is the one that matters before a long run. It makes one
model call and refuses the sweep if the policy and the environment disagree —
on action width, on the prompt's stated limits, on whether the renderer
produced a frame at all.

## Running a sweep

```bash
# One task, five seeds, the model that scores best (see below).
pixi run -e robocasa python -m examples.advanced.robocasa.study_app \
  --methods model --model-id gemini-4-argon \
  --task PickPlaceCounterToStove --seeds 0 1 2 3 4 \
  --official-horizons --preflight --checkpoint out/run.jsonl

# The full leaderboard split: 18 tasks x 50 trials.
pixi run -e robocasa python -m examples.advanced.robocasa.study_app \
  --methods model --model-id gemini-4-argon \
  --atomic-seen --official-horizons --seeds $(seq 0 49) \
  --preflight --checkpoint out/full.jsonl
```

`--checkpoint` makes a run resumable: finished cells are read back and skipped,
so an interrupted sweep costs one episode rather than the run.

An episode is ~600-1050 control steps at roughly 25 minutes wall and 60-140
model calls. Five in parallel is comfortable; **do not run more than about
three per 16 GB of RAM** — each worker peaks near 1.5 GB and eleven at once
took an 18 GB machine to its knees.

## Which model

Eleven were screened on one task and seed. Scores track model generation almost
monotonically, and two results are worth knowing before you choose:
`gemini-2.5-pro` came **last**, below `gemini-2.5-flash`, and the
robotics-specific `gemini-robotics-er-2-preview` placed mid-pack behind four
general models.

| model | score | closest approach | note |
| --- | --- | --- | --- |
| `gemini-4-argon` | 50 | 0.032 m | best; also the cheapest in calls |
| `gemini-3.5-flash` | 25 | 0.026 m | runner-up |
| `gemini-robotics-er-2-preview` | 0 | 0.121 m | 2.4x the calls, 14% refusals |
| `gemini-2.5-flash` | 0 | 0.512 m | |
| `gemini-2.5-pro` | 0 | 0.622 m | last of eleven |

**Ten of the twelve usable models are served only from the `global` Vertex
endpoint** and return 404 in `us-central1`; `gemini-robotics-er-2-preview`
returns a persistent 429 there that reads as a rate limit and is a zero quota.
The client routes these automatically — see `GLOBAL_ONLY_MODELS` in
`study_gemini.py`. A 404 from one location is not evidence a model is
unavailable, which is how the whole 3.x line was first written off here.

## What the numbers currently are

Five seeds each on `PickPlaceCounterToStove`, after three harness fixes:

| model | success | mean score | rungs credited |
| --- | --- | --- | --- |
| `gemini-4-argon` | 0/5 | 30.0 | reached x3, grasped x2, lifted x1 |
| `gemini-3.5-flash` | 0/5 | 15.0 | reached x3 |
| `gemini-robotics-er-2-preview` | 0/5 | 0.0 | none |
| `gemini-2.5-flash` | 0/5 | 0.0 | none |

Success is zero everywhere and the reason is structural rather than mysterious.
The task's own predicate is

```python
check_obj_in_receptacle(obj, container, th=0.07) and gripper_obj_far(th=0.25)
```

so it needs six things in order: reach, grasp, lift, **carry the object within
0.07 m of the pan**, **release so it contacts the pan**, and **withdraw the
gripper 0.25 m**. The study's rung ladder stops at *lifted*, so links four
through six are invisible in every record, and nothing here has ever performed
them — including the privileged scripted reference, whose own notes say what is
left is transport and release. For contrast, a trained VLA
(`lerobot/smolvla_robocasa`) reaches on 46 of 50 episodes of this task, grasps
on 30, and succeeds on 3.

## Three fixes worth knowing about

Each was being measured as a property of the model until it was found, and none
was visible in an episode record — an episode of ineffective actions and one of
misdirected actions both end with the predicate unmet and no rung credited.

1. **Action scaling.** The codec speaks metres; the operational-space
   controller takes a normalised [-1, 1] and multiplies by its own 0.05 m
   per-step maximum. Forty steps asking for 0.800 m moved the arm 0.009 m. A
   regression test now asserts the scaling.
2. **No JSON mode.** About a third of answers arrived wrapped in prose and
   failed to parse, and a parse failure holds the previous action — so a third
   of the policy's decisions were "repeat the last one". Now 0-14%.
3. **The prompt never named the frame.** It listed three position deltas and
   never said which way any pointed. Measured: +x forward, +y left, +z up.

## The VLA arm, which does want a GPU

`--methods vla` runs `lerobot/smolvla_robocasa`. Inference is 2.58 s per control
step on CPU and 0.075 s on an L4 — 45x — but the bottleneck moves to the
renderer rather than disappearing, so end-to-end is about 8.6x. The
`robocasa-gcp` skill under `.claude/skills/` provisions a GPU VM, shards the
work and tears it down; `run_gpu.sh` caps workers by VRAM as well as vCPU,
because nine workers take 19.9 GiB of an L4's 23.0 GiB.

A full 18-task x 50-trial VLA sweep is about 24 hours and $20 of spot GPU.
