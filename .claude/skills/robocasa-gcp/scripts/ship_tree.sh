#!/usr/bin/env bash
# Copy the working tree to a VM. Sourced by provision.sh and build_image.sh.
#
# A tarball of the tree rather than a git clone, because the work is usually on
# an unpushed branch and a clone would silently run different code — the kind of
# difference that produces plausible numbers from the wrong source.

set -euo pipefail
ship_tree() {
  local vm="$1" root="$2"
  local tarball; tarball="$(mktemp -t rctree).tar.gz"
  say "packing the working tree (excluding .pixi, .git, out, node_modules)"
  tar --exclude='.pixi' --exclude='.git' --exclude='out' \
      --exclude='node_modules' --exclude='__pycache__' --exclude='*.pyc' \
      -czf "$tarball" -C "$root" .
  say "tree is $(du -h "$tarball" | cut -f1)"
  gcloud compute ssh "$vm" --zone="$RC_ZONE" --quiet \
    --command="mkdir -p ~/golden-retriever"
  gcloud compute scp "$tarball" "$vm":~/tree.tar.gz --zone="$RC_ZONE" --quiet
  gcloud compute ssh "$vm" --zone="$RC_ZONE" --quiet \
    --command="tar -xzf ~/tree.tar.gz -C ~/golden-retriever && rm ~/tree.tar.gz"
  rm -f "$tarball"
  say "tree delivered to ~/golden-retriever"
}
