#!/usr/bin/env bash
# Shared settings and helpers. Source this; do not run it.
#
# Every value here can be overridden from the environment, so a caller can
# retarget zone, machine type or image without editing a script.

set -euo pipefail

: "${RC_ZONE:=us-central1-a}"
: "${RC_VM:=robocasa-cpu}"
: "${RC_IMAGE_FAMILY:=robocasa-cpu}"
# Physics is 8 ms/step and osmesa is ~45 ms per camera per step, so cores buy
# parallel episodes. 16 vCPU comfortably runs the 6 workers that took a local
# sweep from 7.4 h to 75 min.
: "${RC_MACHINE:=n2-standard-16}"
: "${RC_DISK_GB:=120}"
: "${RC_BUILD_VM:=robocasa-cpu-build}"
: "${RC_BUILD_MACHINE:=n2-standard-8}"

RC_PROJECT="${RC_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
: "${RC_BUCKET:=gs://${RC_PROJECT}-robocasa-runs}"

say()  { printf '  %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die()  { printf '  ERROR: %s\n' "$*" >&2; exit 1; }

require_project() {
  [ -n "$RC_PROJECT" ] || die "no GCP project; run 'gcloud config set project PROJECT'"
}

vm_exists() {
  gcloud compute instances describe "$1" --zone="$RC_ZONE" \
    --format="value(name)" >/dev/null 2>&1
}

# Run a command on the VM. Keeps the ssh invocation in one place so the quiet
# flags and the zone cannot drift between scripts.
on_vm() {
  gcloud compute ssh "$RC_VM" --zone="$RC_ZONE" --quiet --command="$1"
}

# Resolve the newest image in the family. Prints the name on success; on
# failure prints nothing and returns 2 for "could not ask" or 1 for "asked, and
# there is none". Those are different problems with different fixes, and
# collapsing them sends a reader off to rebuild an image when the real cause was
# an expired credential or lapsed billing.
# The reason goes to a file, not a variable: callers use `$(image_in_family)`,
# which runs in a subshell, so anything this assigns is discarded the moment it
# returns. That is how the first version of this reported "unknown error" while
# holding the real one.
RC_IMAGE_ERROR_FILE="${TMPDIR:-/tmp}/rc_image_error.$$"

image_reason() {
  [ -s "$RC_IMAGE_ERROR_FILE" ] && tr '\n' ' ' < "$RC_IMAGE_ERROR_FILE" \
    || printf 'no detail captured'
}

image_in_family() {
  local out status
  out="$(gcloud compute images describe-from-family "$RC_IMAGE_FAMILY" \
         --format='value(name)' 2>&1)"
  status=$?
  if [ "$status" -eq 0 ] && [ -n "$out" ]; then
    printf '%s' "$out"
    return 0
  fi
  printf '%s' "$out" > "$RC_IMAGE_ERROR_FILE"
  case "$out" in
    *"was not found"*|*"NOT_FOUND"*) return 1 ;;   # asked, and there is none
    *) return 2 ;;                                 # could not ask
  esac
}

ensure_bucket() {
  if ! gcloud storage buckets describe "$RC_BUCKET" >/dev/null 2>&1; then
    say "creating results bucket $RC_BUCKET"
    gcloud storage buckets create "$RC_BUCKET" --location=US --quiet
  fi
}

# The environment every remote command needs. osmesa alone is not enough:
# PyOpenGL needs its own platform variable or it fails with a NoneType error
# that names neither.
remote_env() {
  printf '%s' 'export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE'
}
