"""
TWIST2 MuJoCo Web demo.

The browser owns MuJoCo physics through mujoco-js. Retriever owns the motion
player and ONNX policy, then streams policy actions back over a WebSocket.
"""

import argparse
import re
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from retriever.flow import Flow, Latest, Pipeline, Rate, io

TWIST2_SIM_DIR = Path(__file__).resolve().parents[1] / "twist2_simulation"
if str(TWIST2_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(TWIST2_SIM_DIR))

from examples.advanced.twist2_simulation.app import _ensure_required_assets
from examples.advanced.twist2_simulation.flows import (
    EnvOutput,
    MotionPlayerFlow,
    PolicyOutput,
    Twist2PolicyFlow,
    quatToEuler,
)

NUM_ACTIONS = 29
DEFAULT_DOF_POS = np.array(
    [
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.4,
        0.0,
        1.2,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.4,
        0.0,
        1.2,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
ACTION_SCALE = np.full(NUM_ACTIONS, 0.5, dtype=np.float32)
STIFFNESS = np.array(
    [
        100,
        100,
        100,
        150,
        40,
        40,
        100,
        100,
        100,
        150,
        40,
        40,
        150,
        150,
        150,
        40,
        40,
        40,
        40,
        4.0,
        4.0,
        4.0,
        40,
        40,
        40,
        40,
        4.0,
        4.0,
        4.0,
    ],
    dtype=np.float32,
)
DAMPING = np.array(
    [
        2,
        2,
        2,
        4,
        2,
        2,
        2,
        2,
        2,
        4,
        2,
        2,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        0.2,
        0.2,
        0.2,
        5,
        5,
        5,
        5,
        0.2,
        0.2,
        0.2,
    ],
    dtype=np.float32,
)
TORQUE_LIMITS = np.array(
    [
        100,
        100,
        100,
        150,
        40,
        40,
        100,
        100,
        100,
        150,
        40,
        40,
        150,
        150,
        150,
        40,
        40,
        40,
        40,
        4.0,
        4.0,
        4.0,
        40,
        40,
        40,
        40,
        4.0,
        4.0,
        4.0,
    ],
    dtype=np.float32,
)
ANKLE_IDX = [4, 5, 10, 11]


@io
@dataclass
class BrowserTwist2State:
    time: float | None = None
    qpos: list[float] | None = None
    qvel: list[float] | None = None
    last_action: list[float] | None = None


_state_lock = threading.Lock()
_policy_lock = threading.Lock()
_latest_state = BrowserTwist2State()
_latest_state_at = 0.0
_latest_policy = PolicyOutput(policy_action=np.zeros(NUM_ACTIONS, dtype=np.float32))
_latest_policy_at = 0.0
_xml_path: Path | None = None
_mesh_dir: Path | None = None
_mesh_files: list[str] = []


app = FastAPI(title="Retriever TWIST2 MuJoCo Web")


def _mesh_files_from_xml(xml_path: Path) -> list[str]:
    xml_text = xml_path.read_text(encoding="utf-8")
    files = list(dict.fromkeys(re.findall(r'<mesh[^>]*file="([^"]+)"', xml_text)))
    return files


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = Path(__file__).with_name("static") / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.get("/model.xml")
async def model_xml():
    if _xml_path is None:
        raise HTTPException(status_code=503, detail="TWIST2 XML path is not configured")
    return Response(_xml_path.read_text(encoding="utf-8"), media_type="application/xml")


@app.get("/asset-manifest.json")
async def asset_manifest():
    return {"mesh_files": _mesh_files}


@app.get("/assets/g1/meshes/{filename}")
async def mesh_asset(filename: str):
    if _mesh_dir is None:
        raise HTTPException(status_code=503, detail="TWIST2 mesh directory is not configured")
    if filename not in _mesh_files:
        raise HTTPException(status_code=404, detail="Mesh is not referenced by the active XML")
    mesh_path = _mesh_dir / filename
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="Mesh file is missing")
    return FileResponse(mesh_path)


@app.get("/constants.json")
async def constants():
    return {
        "num_actions": NUM_ACTIONS,
        "default_dof_pos": DEFAULT_DOF_POS.astype(float).tolist(),
        "action_scale": ACTION_SCALE.astype(float).tolist(),
        "stiffness": STIFFNESS.astype(float).tolist(),
        "damping": DAMPING.astype(float).tolist(),
        "torque_limits": TORQUE_LIMITS.astype(float).tolist(),
    }


@app.websocket("/ws/policy")
async def websocket_policy(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            state = BrowserTwist2State(
                time=message.get("time"),
                qpos=message.get("qpos"),
                qvel=message.get("qvel"),
                last_action=message.get("last_action"),
            )

            global _latest_state, _latest_state_at
            with _state_lock:
                _latest_state = state
                _latest_state_at = time.time()

            with _policy_lock:
                policy_action = np.asarray(_latest_policy.policy_action, dtype=np.float32)
                policy_age_ms = (time.time() - _latest_policy_at) * 1000.0

            await websocket.send_json(
                {
                    "policy_action": policy_action.astype(float).tolist(),
                    "policy_age_ms": policy_age_ms,
                    "server_time": time.time(),
                }
            )
    except WebSocketDisconnect:
        pass


class BrowserStateSource(Flow[None, BrowserTwist2State]):
    def reset(self) -> None:
        self.period_s = 1.0 / 50.0

    def step(self, _input: None) -> BrowserTwist2State:
        time.sleep(self.period_s)
        with _state_lock:
            return _latest_state


class WebProprioFlow(Flow[BrowserTwist2State, EnvOutput]):
    def reset(self) -> None:
        self.last_log_at = 0.0

    def step(self, state: BrowserTwist2State) -> EnvOutput:
        if state is None or state.qpos is None or state.qvel is None:
            return EnvOutput(proprio=None, vis=None)

        qpos = np.asarray(state.qpos, dtype=np.float32)
        qvel = np.asarray(state.qvel, dtype=np.float32)
        if qpos.shape[0] < 7 + NUM_ACTIONS or qvel.shape[0] < 6 + NUM_ACTIONS:
            return EnvOutput(proprio=None, vis=None)

        dof_pos = qpos[7 : 7 + NUM_ACTIONS]
        dof_vel = qvel[6 : 6 + NUM_ACTIONS]
        quat = qpos[3:7]
        ang_vel = qvel[3:6]

        if state.last_action is not None and len(state.last_action) == NUM_ACTIONS:
            last_action = np.asarray(state.last_action, dtype=np.float32)
        else:
            with _policy_lock:
                last_action = np.asarray(_latest_policy.policy_action, dtype=np.float32)

        obs_body_dof_vel = dof_vel.copy()
        obs_body_dof_vel[ANKLE_IDX] = 0.0
        obs_proprio = np.concatenate(
            [
                ang_vel * 0.25,
                quatToEuler(quat)[:2],
                dof_pos - DEFAULT_DOF_POS,
                obs_body_dof_vel * 0.05,
                last_action,
            ]
        ).astype(np.float32)

        now = time.time()
        if now - self.last_log_at >= 1.0:
            self.last_log_at = now
            state_age_ms = (now - _latest_state_at) * 1000.0
            root_z = float(qpos[2])
            print(
                f"[WebProprio] sim_t={state.time or 0.0:.2f}s "
                f"root_z={root_z:.3f} state_age={state_age_ms:.1f}ms"
            )

        return EnvOutput(proprio=obs_proprio, vis=None)


class WebPolicySink(Flow[PolicyOutput, None]):
    def step(self, policy: PolicyOutput) -> None:
        if policy is None or policy.policy_action is None:
            return

        global _latest_policy, _latest_policy_at
        with _policy_lock:
            _latest_policy = policy
            _latest_policy_at = time.time()


def _start_server(host: str, port: int):
    thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={
            "host": host,
            "port": port,
            "log_level": "warning",
            "use_colors": False,
            "log_config": None,
        },
        daemon=True,
    )
    thread.start()
    return thread


def _wait_for_server(host: str, port: int, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"FastAPI server did not start on {host}:{port}")


def main():
    parser = argparse.ArgumentParser(description="TWIST2 MuJoCo Web + Retriever demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--xml", type=str, default="assets/g1/g1_sim2sim_29dof.xml")
    parser.add_argument("--policy", type=str, default="assets/ckpts/twist2_1017_20k.onnx")
    parser.add_argument("--motion", type=str, default="assets/example_motions/0807_yanjie_walk_001.pkl")
    parser.add_argument("--asset-root", type=str, default="assets/twist2")
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--backend",
        default="in-process",
        choices=["in-process"],
        help="The WebSocket bridge shares memory with Retriever flows, so this demo uses in-process.",
    )
    args = parser.parse_args()

    global _xml_path, _mesh_dir, _mesh_files
    asset_root = Path(args.asset_root).expanduser().resolve()
    xml_path, policy_path, motion_path = _ensure_required_assets(
        xml=args.xml,
        policy=args.policy,
        motion=args.motion,
        asset_root=asset_root,
        auto_download=not args.no_auto_download,
    )
    _xml_path = xml_path
    _mesh_dir = xml_path.parent / "meshes"
    _mesh_files = _mesh_files_from_xml(xml_path)

    pipe = Pipeline("twist2_web_demo")
    with pipe:
        browser_state = BrowserStateSource() @ Rate(50)
        proprio = WebProprioFlow() @ Rate(50)
        motion = MotionPlayerFlow(motion_file=str(motion_path)) @ Rate(50)
        policy = Twist2PolicyFlow(policy_path=str(policy_path), device=args.device) @ Rate(50)
        policy_sink = WebPolicySink() @ Rate(50)

        browser_state.then(proprio, sync=Latest())
        proprio.then(policy, sync=Latest())
        motion.then(policy, sync=Latest())
        policy.then(policy_sink, sync=Latest())

    _start_server(args.host, args.port)
    _wait_for_server(args.host, args.port)
    print("TWIST2 MuJoCo Web + Retriever policy demo")
    print(f"  Open:    http://{args.host}:{args.port}")
    print(f"  XML:     {xml_path}")
    print(f"  Policy:  {policy_path}")
    print(f"  Motion:  {motion_path}")
    print(f"  Meshes:  {len(_mesh_files)} files")
    print("  Physics: browser MuJoCo WASM")
    print("  Policy:  Retriever Python ONNX flow @ 50 Hz")
    print("  Stop:    Ctrl+C", flush=True)

    try:
        if args.duration is None:
            pipe.run(backend=args.backend, blocking=True)
        else:
            pipe.run(backend=args.backend, duration=args.duration, blocking=True)
    except KeyboardInterrupt:
        print("\nStopping TWIST2 web demo...")


if __name__ == "__main__":
    main()
