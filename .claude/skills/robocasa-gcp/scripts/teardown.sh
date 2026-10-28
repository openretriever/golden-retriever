#!/usr/bin/env bash
# Delete the run VM. Safe to run twice, and safe after a preemption.
#
# Always run this. A spot VM left behind is the realistic way to spend a
# research credit on nothing at all.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/common.sh"
require_project

step "tearing down"
if vm_exists "$RC_VM"; then
  say "final mirror before deleting"
  on_vm "cd ~/golden-retriever && gcloud storage rsync -q out '$RC_BUCKET/last-out' 2>/dev/null || true" || true
  gcloud compute instances delete "$RC_VM" --zone="$RC_ZONE" --quiet
  say "deleted $RC_VM"
else
  say "$RC_VM does not exist (already gone, or preempted and self-deleted)"
fi

left="$(gcloud compute instances list --filter='labels.purpose=robocasa-run' --format='value(name,status)' 2>/dev/null)"
if [ -z "$left" ]; then say "no robocasa run VMs remain"; else say "STILL RUNNING:"; printf '  %s\n' "$left"; fi

orphans="$(gcloud compute disks list --filter='-users:*' --format='value(name,sizeGb)' 2>/dev/null)"
if [ -z "$orphans" ]; then say "no orphan disks"; else say "orphan disks still billing:"; printf '  %s\n' "$orphans"; fi
