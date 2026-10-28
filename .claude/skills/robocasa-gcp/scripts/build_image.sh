#!/usr/bin/env bash
# Prepare the environment once and capture it as a reusable disk image.
#
# Run this when the image does not exist yet, or when the robosuite/robocasa
# pins or the asset set change. Everything slow lives here: the pixi solve and
# ~13 GB of kitchen assets. Per-run scripts boot from the result.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/common.sh"
. "$HERE/ship_tree.sh"
require_project

IMAGE="${RC_IMAGE_FAMILY}-$(date -u +%Y%m%d)"
ROOT="${RC_REPO:-$(cd "$HERE/../../../.." && pwd)}"

step "building image $IMAGE in $RC_PROJECT / $RC_ZONE"
say "source tree: $ROOT"
say "this takes roughly 30-45 minutes, almost all of it asset download"

if gcloud compute images describe "$IMAGE" --format="value(name)" >/dev/null 2>&1; then
  say "image $IMAGE already exists; delete it first to rebuild"
  exit 0
fi

step "creating the build VM (on demand, not spot: a preemption here wastes the whole build)"
if ! vm_exists "$RC_BUILD_VM"; then
  gcloud compute instances create "$RC_BUILD_VM" \
    --zone="$RC_ZONE" --machine-type="$RC_BUILD_MACHINE" \
    --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
    --boot-disk-size="${RC_DISK_GB}GB" --boot-disk-type=pd-balanced \
    --scopes=cloud-platform --labels=purpose=robocasa-image-build \
    --format="value(name,status)" | sed 's/^/  /'
  say "waiting for ssh"
  for _ in $(seq 1 30); do
    gcloud compute ssh "$RC_BUILD_VM" --zone="$RC_ZONE" --quiet \
      --command="true" >/dev/null 2>&1 && break
    sleep 10
  done
fi

step "shipping the tree and bootstrapping"
RC_VM="$RC_BUILD_VM" ship_tree "$RC_BUILD_VM" "$ROOT"
gcloud compute scp "$HERE/bootstrap.sh" "$RC_BUILD_VM":~/bootstrap.sh \
  --zone="$RC_ZONE" --quiet
gcloud compute ssh "$RC_BUILD_VM" --zone="$RC_ZONE" --quiet \
  --command="bash ~/bootstrap.sh"

step "capturing the image"
gcloud compute instances stop "$RC_BUILD_VM" --zone="$RC_ZONE" --quiet
gcloud compute images create "$IMAGE" \
  --source-disk="$RC_BUILD_VM" --source-disk-zone="$RC_ZONE" \
  --family="$RC_IMAGE_FAMILY" \
  --description="RoboCasa CPU-only: osmesa GL, pixi robocasa env, kitchen assets" \
  --format="value(name,diskSizeGb)" | sed 's/^/  /'

step "deleting the build VM"
gcloud compute instances delete "$RC_BUILD_VM" --zone="$RC_ZONE" --quiet
say "image $IMAGE ready in family $RC_IMAGE_FAMILY"
say "per-run cost is now ~1 minute to boot instead of ~40 to prepare"
