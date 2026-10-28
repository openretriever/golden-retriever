#!/usr/bin/env bash
# Pull results and aggregate them. Works against GCS, so it is useful even if
# the VM has already been preempted or destroyed.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/common.sh"
require_project

WATCH=no
RUN_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --watch) WATCH=yes; shift;;
    --run) RUN_ID="$2"; shift 2;;
    *) die "unknown argument $1";;
  esac
done

if [ -z "$RUN_ID" ]; then
  RUN_ID="$(gcloud storage ls "$RC_BUCKET/" 2>/dev/null \
            | awk -F/ '{print $(NF-1)}' | grep -E '^[0-9]{8}T' | sort | tail -1)"
  [ -n "$RUN_ID" ] || die "no runs in $RC_BUCKET; pass --run RUN_ID"
fi
step "run $RUN_ID"

LOCAL="out/gcp/$RUN_ID"
mkdir -p "$LOCAL"

summarise() {
  gcloud storage rsync -q "$RC_BUCKET/$RUN_ID" "$LOCAL" 2>/dev/null || true
  python3 "$HERE/aggregate.py" "$LOCAL"
}

if [ "$WATCH" = yes ]; then
  while :; do
    step "run $RUN_ID (watching; ctrl-c to stop)"
    summarise
    sleep 60
  done
else
  summarise
  say "local copy: $LOCAL"
fi
