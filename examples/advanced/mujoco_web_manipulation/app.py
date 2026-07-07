import argparse
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

import retriever
from retriever.flow import Flow, Latest, Pipeline, Rate, flow_io


MODEL_XML = """
<mujoco>
  <option timestep="0.005" integrator="RK4" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba=".9 .8 .7 1"/>

    <body name="target" pos="0.5 0.5 0.08" mocap="true">
      <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
    </body>

    <body name="link1" pos="0 0 0.08">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.05" rgba="0 0.7 0.7 1"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.05" rgba="0.7 0 0.7 1"/>
        <site name="tip" pos="0.5 0 0" size="0.01" rgba="0 0 1 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" gear="1"/>
    <motor name="motor2" joint="joint2" gear="1"/>
  </actuator>
</mujoco>
"""


@flow_io
@dataclass
class BrowserState:
    time: Optional[float] = None
    qpos: Optional[list[float]] = None
    qvel: Optional[list[float]] = None
    tip_pos: Optional[list[float]] = None
    target_pos: Optional[list[float]] = None
    jacobian: Optional[list[float]] = None


@flow_io
@dataclass
class Control:
    ctrl: list[float]


_state_lock = threading.Lock()
_control_lock = threading.Lock()
_latest_state = BrowserState()
_latest_state_at = 0.0
_latest_control = Control(ctrl=[0.0, 0.0])
_latest_control_at = 0.0


app = FastAPI(title="Retriever MuJoCo Web Manipulation")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/model.xml")
async def model_xml():
    return Response(MODEL_XML.strip(), media_type="application/xml")


@app.websocket("/ws/control")
async def websocket_control(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            state = BrowserState(
                time=message.get("time"),
                qpos=message.get("qpos"),
                qvel=message.get("qvel"),
                tip_pos=message.get("tip_pos"),
                target_pos=message.get("target_pos"),
                jacobian=message.get("jacobian"),
            )

            global _latest_state, _latest_state_at
            with _state_lock:
                _latest_state = state
                _latest_state_at = time.time()

            with _control_lock:
                control = list(_latest_control.ctrl)
                control_age_ms = (time.time() - _latest_control_at) * 1000.0

            await websocket.send_json(
                {
                    "ctrl": control,
                    "control_age_ms": control_age_ms,
                    "server_time": time.time(),
                }
            )
    except WebSocketDisconnect:
        pass


class BrowserStateSource(Flow[None, BrowserState]):
    def init(self):
        self.period_s = 1.0 / 50.0

    def run(self, _):
        time.sleep(self.period_s)
        with _state_lock:
            return _latest_state


class RetrieverControllerFlow(Flow[BrowserState, Control]):
    def init(self):
        self.kp_cart = 40.0
        self.kd_joint = 2.0
        self.last_log_at = 0.0

    def run(self, state: BrowserState) -> Control:
        if (
            state is None
            or state.qvel is None
            or state.tip_pos is None
            or state.target_pos is None
            or state.jacobian is None
        ):
            return Control(ctrl=[0.0, 0.0])

        qvel = np.asarray(state.qvel, dtype=np.float64)
        tip_pos = np.asarray(state.tip_pos, dtype=np.float64)
        target_pos = np.asarray(state.target_pos, dtype=np.float64)
        jacobian = np.asarray(state.jacobian, dtype=np.float64).reshape(3, 2)

        err_cart = target_pos - tip_pos
        err_cart[2] = 0.0

        f_cart = self.kp_cart * err_cart
        tau = jacobian.T @ f_cart
        tau -= self.kd_joint * qvel
        tau = np.clip(tau, -20.0, 20.0)

        now = time.time()
        if now - self.last_log_at >= 1.0:
            self.last_log_at = now
            age_ms = (time.time() - _latest_state_at) * 1000.0
            dist = float(np.linalg.norm(err_cart))
            print(
                f"[RetrieverController] sim_t={state.time:.2f}s "
                f"dist={dist:.3f}m state_age={age_ms:.1f}ms tau={tau.round(2).tolist()}"
            )

        return Control(ctrl=tau.astype(float).tolist())


class WebControlSink(Flow[Control, None]):
    def run(self, control: Control):
        global _latest_control, _latest_control_at
        if control is None or control.ctrl is None:
            return None

        with _control_lock:
            _latest_control = control
            _latest_control_at = time.time()
        return None


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
    parser = argparse.ArgumentParser(description="MuJoCo Web + Retriever controller demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--backend",
        default="in-process",
        choices=["in-process"],
        help="The WebSocket bridge shares memory with Retriever flows, so this sample uses in-process.",
    )
    args = parser.parse_args()

    pipe = Pipeline("mujoco_web_manipulation")
    with pipe:
        browser_state = BrowserStateSource() @ Rate(50)
        controller = RetrieverControllerFlow() @ Rate(50)
        control_sink = WebControlSink() @ Rate(50)

        browser_state.then(controller, sync=Latest())
        controller.then(control_sink, sync=Latest())

    _start_server(args.host, args.port)
    _wait_for_server(args.host, args.port)
    print("MuJoCo Web + Retriever controller demo")
    print(f"  Open:    http://{args.host}:{args.port}")
    print("  Physics: browser MuJoCo WASM")
    print("  Control: Retriever Python flow @ 50 Hz")
    print("  Stop:    Ctrl+C", flush=True)

    try:
        if args.duration is None:
            pipe.run(backend=args.backend, blocking=True)
        else:
            pipe.run(backend=args.backend, duration=args.duration, blocking=True)
    except KeyboardInterrupt:
        print("\nStopping MuJoCo Web demo...")


if __name__ == "__main__":
    main()
