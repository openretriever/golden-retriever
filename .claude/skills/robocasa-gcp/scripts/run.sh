#!/usr/bin/env bash
# Shard a task list across workers on the VM and mirror results to GCS.
#
# The sweep is latency-bound on the model API, not CPU-bound: a local run sat at
# 0% CPU waiting on calls, and six-way parallelism took it from 7.4 h to 75 min.
# So workers here buy wall-clock almost linearly until the longest single task
# becomes the floor.
#
# Checkpoints are mirrored to GCS as they are written. Spot VMs are preemptible
# and local disk dies with them; the sweep already checkpoints per episode, so
# mirroring makes a preemption cost one episode instead of a run.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/common.sh"
require_project
vm_exists "$RC_VM" || die "$RC_VM does not exist; run provision.sh"

WORKERS=6
RESUME=""
PASSTHROUGH=()
while [ $# -gt 0 ]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2;;
    --resume)  RESUME="$2"; shift 2;;
    *) PASSTHROUGH+=("$1"); shift;;
  esac
done
[ "${#PASSTHROUGH[@]}" -gt 0 ] || die "pass the study_app arguments, e.g. --atomic-seen --methods model --seeds 0"

# Resuming reuses the run id, which is what makes the mirrored checkpoints
# findable. Without this the mirror preserved data that nothing restored, and
# a lost VM meant a lost run rather than a lost episode.
if [ -n "$RESUME" ]; then
  RUN_ID="$RESUME"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
step "run $RUN_ID: $WORKERS workers, args: ${PASSTHROUGH[*]}"
ensure_bucket

# The worker script is generated here so the argument list and the run id are
# fixed in one place and cannot drift between the launcher and the mirror.
cat > /tmp/rc_worker.sh <<WORKER
#!/usr/bin/env bash
set -uo pipefail
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE
cd "\$HOME/golden-retriever"
export PATH="\$HOME/.pixi/bin:\$PATH"
i="\$1"; shift
mkdir -p "out/$RUN_ID"
# --checkpoint makes a worker resumable, so a preemption costs one episode.
pixi run -e robocasa python -m examples.advanced.robocasa.study_app \\
  "\$@" --checkpoint "out/$RUN_ID/w\$i.jsonl" \\
  > "out/$RUN_ID/w\$i.json" 2> "out/$RUN_ID/w\$i.err"
echo "worker \$i exit \$?" >> "out/$RUN_ID/status"
WORKER

cat > /tmp/rc_mirror.sh <<MIRROR
#!/usr/bin/env bash
# Copy results off-box every 60s for as long as workers are running.
set -uo pipefail
cd "\$HOME/golden-retriever"
while pgrep -f "study_app" >/dev/null; do
  gcloud storage rsync -q "out/$RUN_ID" "$RC_BUCKET/$RUN_ID" 2>/dev/null || true
  sleep 60
done
gcloud storage rsync -q "out/$RUN_ID" "$RC_BUCKET/$RUN_ID" 2>/dev/null || true
echo done > "out/$RUN_ID/mirrored"
MIRROR

gcloud compute scp /tmp/rc_worker.sh /tmp/rc_mirror.sh "$RC_VM":~/ \
  --zone="$RC_ZONE" --quiet
rm -f /tmp/rc_worker.sh /tmp/rc_mirror.sh

if [ -n "$RESUME" ]; then
  step "restoring checkpoints for $RUN_ID from $RC_BUCKET"
  on_vm "cd ~/golden-retriever && mkdir -p 'out/$RUN_ID' && \
    gcloud storage rsync -q '$RC_BUCKET/$RUN_ID' 'out/$RUN_ID' 2>/dev/null || true; \
    echo \"  episodes already complete: \$(cat out/$RUN_ID/w*.jsonl 2>/dev/null | grep -c . || echo 0)\""
  say "the sweep skips any cell already in a checkpoint, so only the rest runs"
fi

# Tasks are split longest-first so the workers finish together rather than one
# trailing by half an hour.
step "sharding tasks across $WORKERS workers (longest first)"
on_vm "export PATH=\$HOME/.pixi/bin:\$PATH; cd ~/golden-retriever && \
  PYTHONPATH=. pixi run -e robocasa python - <<'PY'
import json, subprocess, sys
from examples.advanced.robocasa.study_tasks import ATOMIC_SEEN_TASKS, official_horizon
tasks = list(ATOMIC_SEEN_TASKS)
groups = [[] for _ in range($WORKERS)]; load = [0]*$WORKERS
for task in sorted(tasks, key=lambda t: -official_horizon(t)):
    i = load.index(min(load)); groups[i].append(task); load[i] += official_horizon(task)
for i, (g, l) in enumerate(zip(groups, load)):
    print(f'  worker {i}: {l:5d} steps  {\" \".join(g)}')
json.dump(groups, open('/tmp/groups.json','w'))
PY"

step "launching"
# Arguments go to the VM in a file rather than through nested shell quoting,
# which is the kind of thing that silently drops a flag.
printf '%s\n' "${PASSTHROUGH[@]}" > /tmp/rc_args.txt
gcloud compute scp /tmp/rc_args.txt "$RC_VM":~/rc_args.txt --zone="$RC_ZONE" --quiet
rm -f /tmp/rc_args.txt

on_vm "cd ~/golden-retriever && python3 - <<'PYEOF'
import json, pathlib, subprocess
groups = json.load(open('/tmp/groups.json'))
args = [l for l in pathlib.Path.home().joinpath('rc_args.txt').read_text().splitlines() if l]
launched = 0
for i, group in enumerate(groups):
    if not group:
        continue
    cmd = ['nohup', 'bash', str(pathlib.Path.home() / 'rc_worker.sh'), str(i)] + args
    for task in group:
        cmd += ['--task', task]
    subprocess.Popen(cmd, stdout=open(f'/tmp/w{i}.log', 'w'),
                     stderr=subprocess.STDOUT, start_new_session=True)
    launched += 1
print('  launched', launched, 'workers with args:', ' '.join(args))
PYEOF
nohup bash ~/rc_mirror.sh >/dev/null 2>&1 & echo '  mirror started'"

say "run id: $RUN_ID"
say "results mirror to $RC_BUCKET/$RUN_ID every 60s"
say "watch:   bash $HERE/collect.sh --watch"
