#!/usr/bin/env bash
# Create a spot VM from the prepared image and refresh it with the local tree.
#
# Spot by default: this work is restartable and checkpointed, so paying on-demand
# rates for it is waste. `--instance-termination-action=DELETE` means a
# preemption cleans up after itself rather than leaving a stopped VM billing for
# its disk.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/common.sh"
. "$HERE/ship_tree.sh"
require_project

WORKERS=6
while [ $# -gt 0 ]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2;;
    --machine) RC_MACHINE="$2"; shift 2;;
    --on-demand) SPOT=no; shift;;
    *) die "unknown argument $1";;
  esac
done
: "${SPOT:=yes}"

step "provisioning $RC_VM ($RC_MACHINE, ${WORKERS} workers, spot=$SPOT)"
IMAGE="$(image_in_family)" || case $? in
  1) die "no image in family $RC_IMAGE_FAMILY; run build_image.sh first" ;;
  *) printf '  could not query images, so whether one exists is unknown:\n' >&2
     printf '    %s\n' "$(image_reason)" >&2
     die "fix the above, then re-run; this is not a missing image" ;;
esac
say "booting from $IMAGE"

if vm_exists "$RC_VM"; then
  say "$RC_VM already exists; reusing it"
else
  args=(--zone="$RC_ZONE" --machine-type="$RC_MACHINE"
        --image="$IMAGE" --boot-disk-size="${RC_DISK_GB}GB"
        --boot-disk-type=pd-balanced --scopes=cloud-platform
        --labels=purpose=robocasa-run)
  if [ "$SPOT" = yes ]; then
    args+=(--provisioning-model=SPOT --instance-termination-action=DELETE)
  fi
  gcloud compute instances create "$RC_VM" "${args[@]}" \
    --format="value(name,status,machineType.basename())" | sed 's/^/  /'
  say "waiting for ssh"
  for _ in $(seq 1 30); do
    on_vm "true" >/dev/null 2>&1 && break
    sleep 10
  done
fi

step "refreshing the working tree (the image's copy is as old as the image)"
ship_tree "$RC_VM" "${RC_REPO:-$(cd "$HERE/../../../.." && pwd)}"

ensure_bucket
say "results bucket: $RC_BUCKET"
on_vm "nproc | sed 's/^/  vcpu /'; free -g | awk 'NR==2{print \"  ram \"\$2\" GB\"}'"
say "provisioned; run verify.sh next"
