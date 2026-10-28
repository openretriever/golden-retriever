#!/usr/bin/env bash
# Shard RoboCasa tasks across workers on the VM and mirror results to GCS.
#
# Generic by design: you name the module to run and the arguments it takes, and
# this adds `--task NAME` per shard plus `--checkpoint PATH`. Any module
# accepting those two flags can be driven here. `example_payload.py` in this
# directory is the smallest module that does, and is shipped to the VM so
# `--module example_payload` works out of the box.
#
# Workers buy wall-clock almost linearly when the payload waits on anything
# external. A model-backed sweep measured locally sat at 0% CPU waiting on API
# calls, and six-way parallelism took it from 7.4 h to 75 min. The floor is the
# longest single task.
#
# Checkpoints mirror to GCS as they are written, because spot VMs are
# preemptible and local disk dies with them. `--resume RUN_ID` brings them back.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/common.sh"
require_project
vm_exists "$RC_VM" || die "$RC_VM does not exist; run provision.sh"

WORKERS=6
MODULE=""
RESUME=""
TASKS=()
PASSTHROUGH=()
while [ $# -gt 0 ]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2;;
    --module)  MODULE="$2";  shift 2;;
    --resume)  RESUME="$2";  shift 2;;
    --task)    TASKS+=("$2"); shift 2;;
    --) shift; while [ $# -gt 0 ]; do PASSTHROUGH+=("$1"); shift; done;;
    *) PASSTHROUGH+=("$1"); shift;;
  esac
done
[ -n "$MODULE" ] || die "pass --module, e.g. --module example_payload"

# Default task set: the RoboCasa365 leaderboard's atomic-seen split, which is
# the published list those success rates are reported against. Listed here
# rather than imported so this skill stands alone; override with --task.
if [ "${#TASKS[@]}" -eq 0 ]; then
  TASKS=(CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug
         NavigateKitchen OpenCabinet OpenDrawer OpenStandMixerHead
         PickPlaceCounterToCabinet PickPlaceCounterToStove
         PickPlaceDrawerToCounter PickPlaceSinkToCounter
         PickPlaceToasterToCounter SlideDishwasherRack TurnOffStove
         TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet)
  say "no --task given; using the leaderboard's 18 atomic-seen tasks"
fi

# Resuming reuses the run id, which is what makes the mirrored checkpoints
# findable. Without it the mirror preserved data that nothing restored, and a
# lost VM meant a lost run rather than a lost episode.
if [ -n "$RESUME" ]; then
  RUN_ID="$RESUME"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi

step "run $RUN_ID: $WORKERS workers, ${#TASKS[@]} tasks, module $MODULE"
ensure_bucket

cat > /tmp/rc_worker.sh <<WORKER
#!/usr/bin/env bash
set -uo pipefail
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa KMP_DUPLICATE_LIB_OK=TRUE
# The repo itself, plus this skill's own scripts so the example payload is
# importable by module name.
export PYTHONPATH=".:\$HOME/rc_scripts"
export PATH="\$HOME/.pixi/bin:\$PATH"
cd "\$HOME/golden-retriever"
i="\$1"; shift
mkdir -p "out/$RUN_ID"
pixi run -e robocasa python -m "$MODULE" \\
  "\$@" --checkpoint "out/$RUN_ID/w\$i.jsonl" \\
  > "out/$RUN_ID/w\$i.json" 2> "out/$RUN_ID/w\$i.err"
echo "worker \$i exit \$?" >> "out/$RUN_ID/status"
WORKER

# The stop condition counts finished workers rather than matching a process
# name. `pgrep -f rc_worker.sh` matches the pgrep itself — its own command line
# contains the pattern — so the mirror never saw zero and ran forever, keeping
# the VM busy after the work was done. Counting lines in the status file each
# worker appends to on exit cannot self-match.
cat > /tmp/rc_mirror.sh <<MIRROR
#!/usr/bin/env bash
set -uo pipefail
cd "\$HOME/golden-retriever"
expected=$WORKERS
for _ in \$(seq 1 480); do
  gcloud storage rsync -q "out/$RUN_ID" "$RC_BUCKET/$RUN_ID" 2>/dev/null || true
  done_count=\$(grep -c . "out/$RUN_ID/status" 2>/dev/null || echo 0)
  [ "\$done_count" -ge "\$expected" ] && break
  sleep 60
done
gcloud storage rsync -q "out/$RUN_ID" "$RC_BUCKET/$RUN_ID" 2>/dev/null || true
echo done > "out/$RUN_ID/mirrored"
MIRROR

printf '%s\n' "${PASSTHROUGH[@]}" > /tmp/rc_args.txt
printf '%s\n' "${TASKS[@]}" > /tmp/rc_tasks.txt
gcloud compute scp /tmp/rc_worker.sh /tmp/rc_mirror.sh /tmp/rc_args.txt /tmp/rc_tasks.txt \
  "$RC_VM":~/ --zone="$RC_ZONE" --quiet
on_vm "mkdir -p ~/rc_scripts"
gcloud compute scp "$HERE/example_payload.py" "$RC_VM":~/rc_scripts/ \
  --zone="$RC_ZONE" --quiet
rm -f /tmp/rc_worker.sh /tmp/rc_mirror.sh /tmp/rc_args.txt /tmp/rc_tasks.txt

if [ -n "$RESUME" ]; then
  step "restoring checkpoints for $RUN_ID from $RC_BUCKET"
  on_vm "cd ~/golden-retriever && mkdir -p 'out/$RUN_ID' && \
    gcloud storage rsync -q '$RC_BUCKET/$RUN_ID' 'out/$RUN_ID' 2>/dev/null || true; \
    echo \"  episodes already complete: \$(cat out/$RUN_ID/w*.jsonl 2>/dev/null | grep -c . || echo 0)\""
  say "the payload skips any task already in a checkpoint, so only the rest runs"
fi

# Longest task first so the workers finish together rather than one trailing by
# half an hour. Horizons come from robocasa's own registry, which is where its
# evaluation code reads them.
step "sharding $WORKERS ways, longest task first"
on_vm "export PATH=\$HOME/.pixi/bin:\$PATH; cd ~/golden-retriever && \
  PYTHONPATH=. pixi run -e robocasa python - <<'PYEOF'
import json, pathlib
from robocasa.utils.dataset_registry_utils import get_task_horizon
tasks = [t for t in pathlib.Path.home().joinpath('rc_tasks.txt').read_text().split() if t]
def horizon(task):
    try:
        return int(get_task_horizon(task))
    except Exception:
        return 1000  # unknown task: assume a middling budget rather than fail
groups = [[] for _ in range($WORKERS)]
load = [0] * $WORKERS
for task in sorted(tasks, key=lambda t: -horizon(t)):
    i = load.index(min(load)); groups[i].append(task); load[i] += horizon(task)
for i, (group, total) in enumerate(zip(groups, load)):
    if group:
        print(f'  worker {i}: {total:6d} steps  ' + ' '.join(group))
json.dump(groups, open('/tmp/groups.json', 'w'))
PYEOF
"

step "launching"
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
print('  launched', launched, 'workers with args:', ' '.join(args) or '(none)')
PYEOF
nohup bash ~/rc_mirror.sh >/dev/null 2>&1 & echo '  mirror started'"

say "run id: $RUN_ID"
say "results mirror to $RC_BUCKET/$RUN_ID every 60s"
say "watch:   bash $HERE/collect.sh --watch"
