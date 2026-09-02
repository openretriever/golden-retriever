"""Serve the drawer-dash scene live in a browser through mjviser.

the drawer-dash example's `viewer.py` talks to viser directly because mjviser needs mujoco>=3.6
while robosuite's controllers are pinned to 3.3.1. That pin turns out not to
bind here: drawer-dash uses robosuite only for the PandaOmron *model XML*, and
does its own position control in `arm_control.py`. So on mujoco 3.12 mjviser
works, and this serves the same scene and the same routine through it.

    PYTHONPATH=<worktree> python serve_mjviser.py [--port 8090]

The routine is the drawer-dash example's `Choreography`, stepped from mjviser's step_fn. Nothing
about the scene or the choreography is modified.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mujoco
import mjviser

from examples.advanced.robocasa_drawer_dash import plan, sequence
from examples.advanced.robocasa_drawer_dash.arm_control import Arm
from examples.advanced.robocasa_drawer_dash.plan import PHASES, TOTAL_SECONDS
from examples.advanced.robocasa_drawer_dash.scene import ensure_scene
from examples.advanced.robocasa_drawer_dash.sequence import Choreography

# The stock slot sits 0.24 m behind the handle, which puts the jar close enough
# to the drawer front that the arm clips it on the way down: measured, the
# drawer is shoved from 0.403 m back to 0.300 m across "lower into drawer" and
# "let go of jar", and stays there for the rest of the cycle. Placing 0.30 m
# behind the handle and releasing at 1.00 keeps it at 0.403 m untouched.
# Applied as a runtime override so the drawer-dash example's plan.py stays as written.
SLOT_PLACE_Z, HANDLE_TO_INTERIOR = 1.00, 0.30
for _m in (plan, sequence):
    _m.SLOT_PLACE_Z = SLOT_PLACE_Z
    _m.HANDLE_TO_INTERIOR = HANDLE_TO_INTERIOR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--dash-port", type=int, default=8091)
    ap.add_argument("--scene", default=None)
    args = ap.parse_args()

    scene_path = ensure_scene(Path(args.scene) if args.scene else None)
    xml = scene_path.read_text()
    # Same courtesy as the offscreen renderer: the scene caps its offscreen
    # buffer at 1200x800, which mjviser's own capture path can trip over.
    xml = re.sub(r'offwidth="\d+"', 'offwidth="1920"', xml)
    xml = re.sub(r'offheight="\d+"', 'offheight="1080"', xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    arm = Arm(model, data)
    routine = Choreography(model, data, arm)

    # The routine loops. Without a reset at the end of a cycle the second pass
    # reaches for a jar that is already in the drawer and closes on empty air,
    # so recycle the scene each time the choreography comes round.
    elapsed = {"t": 0.0}

    def step_fn(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        routine.step()
        mujoco.mj_step(m, d)
        elapsed["t"] += m.opt.timestep
        if elapsed["t"] >= TOTAL_SECONDS:
            elapsed["t"] = 0.0
            mujoco.mj_resetDataKeyframe(m, d, 0)
            mujoco.mj_forward(m, d)
            routine.reset()

    def reset_fn(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        routine.reset()
        elapsed["t"] = 0.0

    server = __import__("viser").ViserServer(port=args.port)
    viewer = mjviser.Viewer(model, data, step_fn=step_fn, reset_fn=reset_fn, server=server)

    # RoboSuite convention: geom group 0 is collision, 1 and 2 are visual. mjviser
    # shows 0-2 by default and draws convex hulls on top, which paints the whole
    # scene in translucent red. Show the visual groups only — the same choice
    # viewer.py and the offscreen renderer make.
    viewer.scene.geom_groups_visible = [False, True, True, False, False, False]
    viewer.scene.show_convex_hull = False
    viewer.scene.rebuild_visual_handles()
    viewer.scene.request_update()

    # Live readouts. The drawer slides are passive, so "holding handle" is the
    # whole story: when it says no, the drawer stops moving.
    with server.gui.add_folder("retriever"):
        errand = server.gui.add_text("errand", "put the cinnamon away", disabled=True)
        stage = server.gui.add_text("skill", PHASES[0].label, disabled=True)
        opened = server.gui.add_text("drawer out", "0.000 m", disabled=True)
        holding = server.gui.add_text("holding handle", "no", disabled=True)
        jar = server.gui.add_text("holding jar", "no", disabled=True)

    @server.on_client_connect
    def _(client) -> None:
        print(f"  client connected: {client.client_id}")

    port = server.get_port()
    print(f"  scene: {scene_path}")
    print(f"  routine: {len(PHASES)} phases, {TOTAL_SECONDS:.1f}s per cycle")
    print(f"  mujoco {mujoco.__version__} + mjviser")
    print(f"  MJVISER URL: http://localhost:{port}", flush=True)

    # A tiny sidecar so the Retriever dashboard can sit beside the viewer:
    # /state is the live readout, / is a wrapper page with the dashboard on the
    # LEFT and the viser canvas in an iframe on the right. viser pins its own
    # GUI to the right and cannot be moved, hence the wrapper.
    import http.server
    import json as _json
    import socketserver
    import threading
    import time

    HERE = Path(__file__).resolve().parent
    live = {"phase": PHASES[0].label, "mode": PHASES[0].mode, "drawer_open": 0.0,
            "gripping_handle": False, "holding_jar": False, "grip_cmd": 0.04,
            "torso_cmd": 0.0, "progress": 0.0, "stowed": False, "viser_port": port}

    class Sidecar(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(HERE), **k)

        def do_GET(self):
            if self.path.startswith("/state"):
                body = _json.dumps(live).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path in ("/", ""):
                self.path = "/dashboard.html"
            return super().do_GET()

        def log_message(self, *_a):
            pass

    class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    side = Threaded(("", args.dash_port), Sidecar)
    threading.Thread(target=side.serve_forever, daemon=True).start()
    print(f"  RETRIEVER DASHBOARD: http://localhost:{args.dash_port}", flush=True)

    def poll() -> None:
        while True:
            try:
                ph = routine.phase
                live.update(phase=ph.label, mode=ph.mode,
                            drawer_open=round(float(routine.drawer_open), 4),
                            gripping_handle=bool(routine.gripping()),
                            holding_jar=bool(routine.holding_jar()),
                            grip_cmd=float(ph.grip), torso_cmd=float(ph.torso),
                            progress=min(1.0, elapsed["t"] / TOTAL_SECONDS))
                if routine.holding_jar():
                    live["seen_jar"] = True
                if live.get("seen_jar") and not routine.holding_jar():
                    live["stowed"] = True
                if elapsed["t"] < 0.2:
                    live["stowed"] = False; live["seen_jar"] = False
                stage.value = routine.phase.label
                opened.value = f"{routine.drawer_open:.3f} m"
                holding.value = "yes" if routine.gripping() else "no"
                jar.value = "yes" if routine.holding_jar() else "no"
            except Exception:
                pass
            time.sleep(0.1)

    threading.Thread(target=poll, daemon=True).start()
    viewer.run()


if __name__ == "__main__":
    main()
