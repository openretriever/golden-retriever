#!/usr/bin/env bash
# Run a VLA sweep on a GPU VM, sharded across workers that share one accelerator.
#
# Separate from run.sh because the bottleneck moves. On CPU, SmolVLA inference is
# 98% of a control step (2.58 s of 2.63 s) and the simulator is free. On an L4 it
# is 75 ms of 307 ms, so the *renderer* becomes the constraint and the GPU sits
# at 16% utilisation with one worker. That inverts the sizing: workers are bought
# to keep the GPU fed, not to spread inference, and the limit is vCPU for
# rendering rather than the accelerator.
#
# Measured on g2-standard-4 + L4, 3 views at 256px, replan every 5 steps:
#
#   device / GL              s/step   inference   render+physics
#   CPU (4 vCPU), osmesa      3.449       3.374            0.075
#   L4, osmesa                0.643       0.075            0.568
#   L4, EGL                   0.307       0.075            0.232
#
# EGL, not osmesa: with a GPU present the rasteriser is hardware, which is 2.1x
# end-to-end. Both produce non-blank frames (pixel std 74.1 and 73.7); osmesa
# stays the CPU path's choice because there is no device to render on.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/common.sh"

WORKERS=4
SEEDS="0 1 2 3 4"
TASKS=vla
RESUME=""
MAXSTEPS=1200
while [ $# -gt 0 ]; do
  case "$1" in
    --workers)   WORKERS="$2"; shift 2;;
    --seeds)     SEEDS="$2"; shift 2;;
    --tasks)     TASKS="$2"; shift 2;;       # vla | atomic
    --max-steps) MAXSTEPS="$2"; shift 2;;
    --resume)    RESUME="$2"; shift 2;;
    *) die "unknown argument $1";;
  esac
done

RUN_ID="${RESUME:-$(date -u +%Y%m%dT%H%M%SZ)}"
case "$TASKS" in
  vla)    TASK_FLAG="--vla-tasks";;
  atomic) TASK_FLAG="--atomic-seen --official-horizons";;
  *) die "--tasks must be vla or atomic";;
esac

step "GPU sweep $RUN_ID ($TASKS, seeds '$SEEDS', $WORKERS workers)"
[ -n "$RESUME" ] && say "resuming $RUN_ID: completed episodes are skipped"

# One worker per shard. Each loads its own copy of the checkpoint — measured at
# ~3.1 GiB of VRAM including activations, so an L4's 22 GiB holds about seven,
# and vCPU for rendering binds first.
cat > /tmp/rcg_worker.sh <<WORKER
#!/usr/bin/env bash
set -uo pipefail
cd "\$HOME/golden-retriever"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE
export PATH="\$HOME/.pixi/bin:\$PATH"
i="\$1"
mkdir -p "out/$RUN_ID"
# --checkpoint is what makes a lost VM cost one episode rather than the run:
# finished cells are read back and skipped on resume.
.pixi/envs/robocasa-gpu/bin/python -m examples.advanced.robocasa.study_app \\
  --methods vla --device cuda --horizon 1 --max-steps $MAXSTEPS \\
  $TASK_FLAG --seeds \$SEEDS_FOR_WORKER \\
  --checkpoint "out/$RUN_ID/w\$i.jsonl" \\
  > "out/$RUN_ID/w\$i.json" 2> "out/$RUN_ID/w\$i.err"
echo "worker \$i exit \$?" >> "out/$RUN_ID/status"
WORKER

# Shard by seed rather than by task: a task's horizon varies by a factor of two
# across the suite, so splitting on tasks leaves workers idle at the tail while
# one grinds through the longest. Seeds are interchangeable in cost.
say "sharding seeds across $WORKERS workers"
SEED_ARR=($SEEDS)
for i in $(seq 0 $((WORKERS-1))); do
  mine=""
  for j in $(seq $i $WORKERS $((${#SEED_ARR[@]}-1))); do mine="$mine ${SEED_ARR[$j]}"; done
  [ -z "$mine" ] && continue
  echo "SEEDS_FOR_WORKER='${mine# }' bash /tmp/rcg_worker.sh $i" >> /tmp/rcg_launch.txt
  say "  worker $i: seeds${mine}"
done

cat > /tmp/rcg_mirror.sh <<MIRROR
#!/usr/bin/env bash
set -uo pipefail
cd "\$HOME/golden-retriever"
expected=$WORKERS
for _ in \$(seq 1 960); do
  gcloud storage rsync -q "out/$RUN_ID" "$RC_BUCKET/$RUN_ID" 2>/dev/null || true
  # Counting finished workers, not matching a process name: pgrep -f matches its
  # own command line and so never reports zero.
  done_count=\$(grep -c . "out/$RUN_ID/status" 2>/dev/null || echo 0)
  [ "\$done_count" -ge "\$expected" ] && break
  sleep 60
done
gcloud storage rsync -q "out/$RUN_ID" "$RC_BUCKET/$RUN_ID" 2>/dev/null || true
echo done > "out/$RUN_ID/mirrored"
MIRROR

gcloud compute scp /tmp/rcg_worker.sh /tmp/rcg_mirror.sh /tmp/rcg_launch.txt \
  "$RC_VM":~/ --zone="$RC_ZONE" --quiet
rm -f /tmp/rcg_worker.sh /tmp/rcg_mirror.sh /tmp/rcg_launch.txt

ensure_bucket
if [ -n "$RESUME" ]; then
  say "pulling prior checkpoints back from $RC_BUCKET/$RUN_ID"
  on_vm "mkdir -p ~/golden-retriever/out/$RUN_ID && \
         gcloud storage rsync -q '$RC_BUCKET/$RUN_ID' ~/golden-retriever/out/$RUN_ID 2>/dev/null || true; \
         cat ~/golden-retriever/out/$RUN_ID/*.jsonl 2>/dev/null | wc -l | sed 's/^/  episodes already complete: /'"
fi

step "starting workers"
on_vm "cd ~/golden-retriever && mkdir -p out/$RUN_ID && rm -f out/$RUN_ID/status && \
       while read -r line; do eval \"nohup \$line >/dev/null 2>&1 &\"; done < ~/rcg_launch.txt; \
       sleep 2; echo '  workers started:' \$(pgrep -fc rcg_worker.sh || echo 0); \
       nohup bash ~/rcg_mirror.sh >/dev/null 2>&1 & echo '  mirror started'"
say "run id $RUN_ID; results mirror to $RC_BUCKET/$RUN_ID every 60s"
say "resume after a preemption with: run_gpu.sh --resume $RUN_ID"
