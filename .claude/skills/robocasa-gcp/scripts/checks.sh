#!/usr/bin/env bash
# The interface checks, run ON the VM. Copied up by verify.sh.
#
# A file rather than inline ssh arguments, and that is the point: inlining these
# means three layers of quoting (shell, then ssh, then python), and when it goes
# wrong the check prints nothing and reports failure. That happened — verify
# exited 1 with no output while the same render worked by hand. A check that can
# fail silently is worse than no check.
#
# Self-contained: robosuite, robocasa and mujoco directly, plus the repo's own
# `robocasa-smoke` task. No experiment-specific code.

set -uo pipefail
. "$HOME/.profile" 2>/dev/null || true
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE
export PATH="$HOME/.pixi/bin:$PATH"
cd "$HOME/golden-retriever"

FAILED=0
note() { printf '  [%s] %s\n' "$1" "$2"; [ "$1" = ok ] || FAILED=$((FAILED + 1)); }

echo "== 1/4 software GL present and selected"
[ -e /usr/lib/x86_64-linux-gnu/libOSMesa.so ] \
  && note ok "libOSMesa present" || note XX "libOSMesa MISSING (apt install libosmesa6)"
[ -e /usr/include/linux/input.h ] \
  && note ok "linux/input.h present" || note XX "linux/input.h MISSING (apt install linux-libc-dev)"
if [ "${MUJOCO_GL:-}" = osmesa ] && [ "${PYOPENGL_PLATFORM:-}" = osmesa ]; then
  note ok "MUJOCO_GL and PYOPENGL_PLATFORM are osmesa"
else
  note XX "GL variables unset (MUJOCO_GL=${MUJOCO_GL:-unset} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-unset})"
fi

echo "== 2/4 mujoco renders offscreen, frame not blank"
RENDER="$(pixi run -e robocasa python - <<'PYEOF' 2>/dev/null | tail -1
import mujoco
model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
  <light pos="0 0 3"/><geom type="plane" size="2 2 .1"/>
  <body pos="0 0 .3"><geom type="box" size=".2 .2 .2" rgba="1 .2 .2 1"/></body>
  <camera name="c" pos="1.2 -1.2 1" xyaxes="0.7 0.7 0 -0.4 0.4 0.8"/>
</worldbody></mujoco>""")
data = mujoco.MjData(model); mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, 128, 128); renderer.update_scene(data, camera="c")
std = float(renderer.render().std())
print(("OK %.1f" if std > 1.0 else "BLANK %.1f") % std)
PYEOF
)"
case "$RENDER" in
  OK*) note ok "mujoco offscreen render (std ${RENDER#OK })" ;;
  BLANK*) note XX "mujoco rendered a blank frame (std ${RENDER#BLANK }) - software GL is not working" ;;
  *) note XX "mujoco render produced no result: ${RENDER:-no output}" ;;
esac

echo "== 3/4 a real RoboCasa kitchen renders"
KITCHEN="$(pixi run -e robocasa python - <<'PYEOF' 2>/dev/null | tail -1
import warnings, time; warnings.filterwarnings("ignore")
import numpy as np, robocasa, robosuite  # noqa: F401  robocasa registers the tasks
from robosuite.controllers import load_composite_controller_config
import os
task = os.environ.get("RC_VERIFY_TASK", "PickPlaceCounterToCabinet")
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
print("%s %.1f %.0f %d" % ("OK" if frame.std() > 1.0 else "BLANK", frame.std(),
                           (time.perf_counter() - started) / 10 * 1000, env.action_dim))
env.close()
PYEOF
)"
case "$KITCHEN" in
  OK*) set -- $KITCHEN; note ok "kitchen render (std $2, $3 ms/step, action_dim $4)" ;;
  BLANK*) note XX "kitchen rendered a blank frame: $KITCHEN" ;;
  *) note XX "kitchen render produced no result: ${KITCHEN:-no output}" ;;
esac

echo "== 4/4 the repo's own simulator smoke test"
if pixi run -e robocasa robocasa-smoke >/dev/null 2>&1; then
  note ok "robocasa-smoke"
else
  note XX "robocasa-smoke failed (run it directly to see why)"
fi

echo "== summary"
if [ "$FAILED" -eq 0 ]; then
  echo "  all checks passed"
  exit 0
fi
echo "  $FAILED check(s) failed; do not start a long run"
exit 1
