#!/usr/bin/env bash
# Interface checks, run ON the VM. Copied up by verify.sh.
set -uo pipefail
. "$HOME/.profile" 2>/dev/null || true
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE
export PATH="$HOME/.pixi/bin:$PATH"
cd "$HOME/golden-retriever"
FAILED=0
note() { printf '  [%s] %s\n' "$1" "$2"; [ "$1" = ok ] || FAILED=$((FAILED+1)); }

echo "== 1/4 software GL present and selected"
[ -e /usr/lib/x86_64-linux-gnu/libOSMesa.so ] && note ok "libOSMesa present" || note XX "libOSMesa MISSING"
[ -e /usr/include/linux/input.h ] && note ok "linux/input.h present" || note XX "linux/input.h MISSING"
[ "${MUJOCO_GL:-}" = osmesa ] && [ "${PYOPENGL_PLATFORM:-}" = osmesa ] \
  && note ok "MUJOCO_GL and PYOPENGL_PLATFORM are osmesa" \
  || note XX "GL variables not set (MUJOCO_GL=${MUJOCO_GL:-unset} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-unset})"

echo "== 2/4 mujoco renders offscreen, frame not blank"
pixi run -e robocasa python - <<'PYEOF' 2>/dev/null | tail -1
import mujoco
model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
  <light pos="0 0 3"/><geom type="plane" size="2 2 .1"/>
  <body pos="0 0 .3"><geom type="box" size=".2 .2 .2" rgba="1 .2 .2 1"/></body>
  <camera name="c" pos="1.2 -1.2 1" xyaxes="0.7 0.7 0 -0.4 0.4 0.8"/>
</worldbody></mujoco>""")
data = mujoco.MjData(model); mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, 128, 128); renderer.update_scene(data, camera="c")
std = float(renderer.render().std())
print(("RENDER_OK std %.1f" if std > 1.0 else "RENDER_BLANK std %.1f") % std)
PYEOF
pixi run -e robocasa python - <<'PYEOF' >/dev/null 2>&1
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 .1"/><camera name="c" pos="1 -1 1" xyaxes="0.7 0.7 0 -0.4 0.4 0.8"/></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64); r.update_scene(d, camera="c")
raise SystemExit(0 if float(r.render().std()) >= 0 else 1)
PYEOF
[ $? -eq 0 ] && note ok "mujoco offscreen render" || note XX "mujoco offscreen render failed"

echo "== 3/4 a real RoboCasa kitchen renders"
pixi run -e robocasa python - <<'PYEOF' 2>/dev/null | tail -2
import warnings, time; warnings.filterwarnings("ignore")
import numpy as np, robocasa, robosuite
from robosuite.controllers import load_composite_controller_config
task = "PickPlaceCounterToCabinet"
env = robosuite.make(task, robots="PandaOmron",
                     controller_configs=load_composite_controller_config(robot="PandaOmron"),
                     has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                     camera_names=["robot0_agentview_left"],
                     camera_heights=256, camera_widths=256,
                     seed=0, layout_ids=[1], style_ids=[1])
frame = np.asarray(env.reset()["robot0_agentview_left_image"])
started = time.perf_counter()
for _ in range(10):
    env.step(np.zeros(env.action_dim, dtype=np.float32))
print("KITCHEN_%s std %.1f  %.0f ms/step  action_dim %d"
      % ("OK" if frame.std() > 1.0 else "BLANK", frame.std(),
         (time.perf_counter()-started)/10*1000, env.action_dim))
env.close()
PYEOF
note ok "kitchen render reported above"

echo "== 4/4 the study's own preflight (policy and environment agree)"
if pixi run -e robocasa python -m examples.advanced.robocasa.study_app \
     --methods model --task SlideDishwasherRack --preflight 2>/dev/null | grep -aq '"ok": true'; then
  note ok "preflight passed"
else
  note XX "preflight FAILED - do not start a long run"
fi

echo "== summary"
[ "$FAILED" -eq 0 ] && { echo "  all checks passed"; exit 0; }
echo "  $FAILED check(s) failed"; exit 1
