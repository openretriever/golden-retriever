# TWIST2 MuJoCo Web

Run the Unitree G1 TWIST2 controller with MuJoCo physics and rendering in the
browser. Retriever remains in Python: it reconstructs proprioception, runs the
motion stream and ONNX policy, and returns actions over a WebSocket.

This browser bridge was originally contributed by Ethan Goodhart and preserved
in commit `2d192d8`. It is intentionally kept as a distinct optional lane from
the native `twist2_simulation` demo.

## Quick start

```bash
pixi run -e twist2 demo-twist2-web
```

Open the printed URL, normally <http://127.0.0.1:8020>. To stop automatically:

```bash
pixi run -e twist2 demo-twist2-web --duration 60
```

The first run downloads the official TWIST2 model, policy, motion, and mesh
assets into the ignored `assets/twist2/` cache. The browser also loads Three.js
and `mujoco-js` from public CDNs, so first use requires network access.

## What runs where

| Surface | Responsibility |
| --- | --- |
| Browser | MuJoCo WASM physics, Three.js rendering, camera controls, and PD torque application |
| Retriever | Motion reference, 92-value proprioception, ONNX policy inference, and typed Flow scheduling at 50 Hz |
| WebSocket | Browser `qpos`/`qvel` state to Retriever; 29 policy actions back to the browser |

Because the browser owns the graphics and physics context, this lane does not
need `mjpython` on macOS. Use the native `demo-twist2` task when testing the
desktop MuJoCo viewer instead.

## Expected output

The terminal should report resolved assets and a compiled five-node graph:

```text
TWIST2 MuJoCo Web + Retriever policy demo
  Open:    http://127.0.0.1:8020
  Meshes:  35 files
  Physics: browser MuJoCo WASM
  Policy:  Retriever Python ONNX flow @ 50 Hz
Compilation complete: 5 nodes, 7 edges, 5 groups
MotionPlayerFlow: Loaded 416 frames at 29.876... FPS
Twist2PolicyFlow: Loaded ...twist2_1017_20k.onnx on CPUExecutionProvider
```

The browser exposes pause, reset, and overview/top/side camera controls. Its
status panel reports the Retriever connection, MuJoCo load state, simulation
time, root height, policy age, and action norm.

## Current verification boundary

Verified locally on macOS:

- automatic download and resolution of 106 official TWIST2 assets,
- motion and ONNX policy loading in the `twist2` Pixi environment,
- browser UI, XML, mesh, and controller-constant endpoints,
- live browser-state to Retriever-policy WebSocket exchange, and
- 29-action policy responses from the five-node Retriever graph.

The current browser rollout loses balance after startup, so this remains a
source-level contributor example rather than a promoted GoldenRetriever docs
lane. Treat controller/model alignment in MuJoCo WASM as the next validation
target; do not interpret a connected status panel alone as locomotion success.

## Options

```bash
pixi run -e twist2 demo-twist2-web --host 127.0.0.1 --port 8020
pixi run -e twist2 demo-twist2-web --device cpu --duration 60
pixi run -e twist2 demo-twist2-web --no-auto-download
```

Use `--xml`, `--policy`, and `--motion` to test a specific matched asset set.
Use `--asset-root` to place the local cache outside the repository.

## Troubleshooting

- **The command is missing:** run it with `-e twist2`; the task belongs to the optional TWIST2 environment.
- **The page stays on loading:** confirm the browser can reach the Three.js and `mujoco-js` CDNs.
- **Retriever stays disconnected:** use the printed host and port and check that `/ws/policy` is reachable.
- **Assets are missing:** allow the first automatic download or pass explicit local paths.
- **The robot falls:** compare the XML, policy, motion, joint ordering, initial pose, and PD constants as one matched controller bundle.
