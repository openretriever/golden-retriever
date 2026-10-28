---
name: robocasa-gcp
description: Run RoboCasa simulation tasks on a CPU-only Google Cloud VM — provision from a prepared image, shard tasks across workers, collect results to GCS, destroy. Use when a RoboCasa sweep is too slow locally, or when an API-model policy run needs parallelism. No GPU.
---

# RoboCasa on Google Cloud, CPU only

RoboCasa's simulator is MuJoCo on CPU, so a cloud run needs vCPUs and software
GL — never an accelerator. Asking for a GPU here is asking for the wrong thing:
it would sit idle while `osmesa` renders on the CPU.

## The measured facts this is built on

Taken from a probe on an `e2-standard-4` spot VM, not from documentation.

| | ms per control step |
| --- | --- |
| physics only | **8.0** |
| 1 camera @128 | 48.3 |
| 2 cameras @128 | 98.1 |
| 2 cameras @256 | **108.2** |

Software rendering is about **92% of the step cost** — roughly 45 ms per
camera — and it scales with the *number* of cameras far more than with
resolution (@128 to @256 cost only 10 ms more). Two consequences shape every
choice below:

* **Physics is cheap on a modest VM.** 8 ms/step beats a local M-series Mac's
  33 ms, which included rendering. The CPU is not the bottleneck; `osmesa` is.
* **Neither matters much for a model-backed run.** The atomic-seen split is
  11,850 control steps, so about **21 minutes of simulation**, against roughly
  **7 hours of API latency**. Optimise for parallel API calls, not for render
  speed. Prefer fewer, larger cameras over more, smaller ones.

## Three Linux-only requirements

None of these are visible from a macOS checkout, and each fails in a way that
does not name its own cause.

1. **`libosmesa6 libosmesa6-dev libglu1-mesa` from apt.** Conda's `mesalib`
   does *not* provide `libOSMesa`; only the system package does.
2. **`MUJOCO_GL=osmesa` *and* `PYOPENGL_PLATFORM=osmesa`.** Setting only the
   first fails with `AttributeError: 'NoneType' object has no attribute
   'glGetError'`, which reads like a MuJoCo bug and is a missing environment
   variable.
3. **`linux-libc-dev build-essential`.** robosuite depends on `evdev`, which
   compiles against `linux/input.h`. Without the headers the robosuite wheel
   build fails and nothing downstream installs.

`verify` asserts all three rather than trusting them.

## Checking the lock covers linux-64, and a trap in doing so

`bootstrap.sh` runs `pixi install -e robocasa`, which installs from the
committed solve rather than re-solving, so the VM gets the same versions as a
local checkout. That only holds if the lock really carries a linux-64 solve for
this environment, and verifying it has a trap worth knowing.

```
pixi list -e robocasa --platform linux-64   ->  0 packages     (misleading)
```

That reads as "no linux-64 solve" and is false. Pixi's lock format v7 **interns
platform names**, so the linux-64 block sits under a key like `p1`. A regex, a
`yaml.safe_load` and the CLI query above all report a platform literally named
`p1`. The reliable check is to read the package URLs and see which platform
they point at: every entry under the interned key is a
`conda-forge/linux-64/...` URL.

Measured on this repo: **osx-arm64 308 packages, linux-64 355**, with the
robosuite and robocasa git pins, mujoco, torch, lerobot and transformers present
under both. So the install on the VM is a download, not a resolve.

## Shape: a prepared image, then create and destroy

Preparing the environment means a pixi solve plus about **23 GB of kitchen
assets**, which is far too slow to repeat per run. So it is done **once** into a
reusable disk image; every run boots from that image, works, and is destroyed.
That keeps create-and-destroy semantics without paying the download each time.

```
build-image   once, or when pins change   ~30-45 min
provision     per run                     ~1 min
verify        per run                     ~1 min
run           per run                     the actual work
collect       per run                     results to GCS and back
teardown      per run                     always, even on failure
```

## First run, for anyone with their own Google Cloud account

Nothing here is shared between users: the image lives in whichever project your
`gcloud` points at, so you build your own once. That means no credentials pass
between people and nobody needs access to anyone else's project.

```bash
S=.claude/skills/robocasa-gcp/scripts

gcloud auth login                            # the CLI
gcloud auth application-default login        # what the policy client uses
gcloud config set project YOUR_PROJECT_ID

bash $S/doctor.sh --enable-apis              # checks, and enables what is missing
bash $S/build_image.sh                       # once, ~45 min
```

`doctor.sh` is the thing to run when something is wrong. It checks gcloud,
login, application default credentials, the project, billing, the three required
APIs, and whether your image exists — and prints the exact command for anything
missing. It changes nothing unless you pass `--enable-apis`.

One check there is less obvious than it looks: it calls
`gcloud auth print-access-token` rather than reading the config, because a
*stale* refresh token still shows a logged-in account while every Vertex call
returns `404 Publisher model ... not found`. That reads as "the model does not
exist" and is really "your credentials expired", and it cost an hour to
diagnose once already.

## Commands

Scripts live in `scripts/`. Every one is idempotent and prints what it will do
before doing it.

```bash
S=.claude/skills/robocasa-gcp/scripts

# Once: prepare the environment and capture it as an image.
bash $S/build_image.sh                      # creates image robocasa-cpu-<date>

# Per run:
bash $S/provision.sh --workers 6            # spot VM sized to the worker count
bash $S/verify.sh                           # GL, two render checks, robocasa-smoke
bash $S/run.sh --workers 6 --module examples.advanced.robocasa.app -- --your-flags
bash $S/collect.sh                          # pull checkpoints, aggregate
bash $S/teardown.sh                         # delete the VM, always
```

`run.sh` shards by task. With no `--task`, it uses the RoboCasa365
leaderboard's eighteen atomic-seen task names, listed in the script rather than
imported so the skill stands alone. The module it runs must accept `--task
NAME` (repeatable) and `--checkpoint PATH`; everything after `--` is passed
through untouched.

## Rules this skill follows

* **Spot instances, with `--instance-termination-action=DELETE`.** A forgotten
  VM is the realistic way to burn a research credit, so every provision carries
  a termination action and `teardown.sh` is safe to run twice.
* **Checkpoints go to GCS as they are written.** Spot VMs are preemptible and
  local disk dies with them. The sweep already checkpoints per episode; this
  mirrors each one off-box so a preemption costs one episode, not a run.
* **No key files.** The VM's attached service account supplies Application
  Default Credentials, so a Vertex-backed policy authenticates with nothing
  written to disk. The VM needs the `cloud-platform` scope and the service
  account needs `roles/aiplatform.user`.
* **The working tree is shipped, not cloned.** Work in progress is usually on an
  unpushed branch, so `provision.sh` sends a tarball of the tree (excluding
  `.pixi`, `.git`, `out`, `node_modules`). A `git clone` would silently run
  different code.
* **The checks live in `checks.sh`, shipped to the VM and run there**, not
  inlined into `ssh --command`. Inlining means three layers of quoting — shell,
  then ssh, then python — and the failure mode is a check that prints nothing
  and exits non-zero while the same code works by hand. That happened on the
  first real run. A check that can fail silently is worse than no check.
* **`verify` gates `run`**, and two of its four checks are render checks that
  assert the frame is not blank. That is the one failure worth catching early: a
  policy handed blank frames looks exactly like a policy that cannot act, and
  nothing raises.
* **Nothing about a particular experiment is baked in.** `run.sh` takes
  `--module` and the arguments that module wants, and adds `--task NAME` per
  shard plus `--checkpoint PATH`. Any module accepting those two flags can be
  driven here, and `verify.sh` uses robosuite, robocasa and mujoco directly so
  it works against a plain checkout.

## What this does not do

GPUs, accelerators, local VLA checkpoints, Isaac Sim or RoboLab-120. RoboLab
needs a GPU for the *simulator*, so no amount of CPU parallelism reaches it.
