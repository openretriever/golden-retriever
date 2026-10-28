#!/usr/bin/env bash
# Check that *this* machine and *this* Google Cloud project can run the skill,
# and say exactly what to do about anything missing.
#
# Everything here is per-user. The image the skill boots from lives in whichever
# project you point gcloud at, so a new user builds their own once rather than
# borrowing someone else's — which also means nobody needs access to anyone
# else's project, and no credentials are shared.
#
# Run this first. It changes nothing unless you pass --enable-apis.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/common.sh"

ENABLE=no
[ "${1:-}" = "--enable-apis" ] && ENABLE=yes

PROBLEMS=0
fail() { printf '  [X] %s\n      -> %s\n' "$1" "$2"; PROBLEMS=$((PROBLEMS + 1)); }
pass() { printf '  [ok] %s\n' "$1"; }

step "checking your setup"

if ! command -v gcloud >/dev/null 2>&1; then
  fail "the gcloud CLI is not installed" \
       "install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
  printf '\n  %s problem(s); nothing else can be checked without gcloud\n' "$PROBLEMS"
  exit 1
fi
pass "gcloud installed ($(gcloud version 2>/dev/null | head -1))"

ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
if [ -z "$ACCOUNT" ] || [ "$ACCOUNT" = "(unset)" ]; then
  fail "not logged in" "run: gcloud auth login"
else
  # A stale refresh token looks like a 404 from Vertex rather than an auth
  # error, so prove the token actually refreshes instead of trusting the config.
  if gcloud auth print-access-token >/dev/null 2>&1; then
    pass "logged in as $ACCOUNT"
  else
    fail "credentials for $ACCOUNT will not refresh" "run: gcloud auth login"
  fi
fi

ADC="$HOME/.config/gcloud/application_default_credentials.json"
if [ -f "$ADC" ]; then
  pass "application default credentials present"
else
  fail "no application default credentials" \
       "run: gcloud auth application-default login  (the policy client uses these, not a key file)"
fi

if [ -z "$RC_PROJECT" ] || [ "$RC_PROJECT" = "(unset)" ]; then
  fail "no project selected" "run: gcloud config set project YOUR_PROJECT_ID"
else
  pass "project $RC_PROJECT"
  BILLED="$(gcloud billing projects describe "$RC_PROJECT" \
            --format='value(billingEnabled)' 2>/dev/null || echo unknown)"
  case "$BILLED" in
    True) pass "billing enabled" ;;
    unknown) fail "could not read billing (needs the Cloud Billing API or permission)" \
                  "check the console: Billing -> Account management" ;;
    *) fail "billing is not enabled on $RC_PROJECT" \
            "a VM cannot be created without it; enable billing in the console" ;;
  esac
fi

step "required APIs"
for api in compute.googleapis.com storage.googleapis.com aiplatform.googleapis.com; do
  if gcloud services list --enabled --format='value(config.name)' 2>/dev/null | grep -qx "$api"; then
    pass "$api"
  elif [ "$ENABLE" = yes ]; then
    say "enabling $api (this can take a minute)"
    if gcloud services enable "$api" --quiet 2>/dev/null; then
      pass "$api enabled"
    else
      fail "could not enable $api" "you may lack permission; ask a project owner"
    fi
  else
    fail "$api is not enabled" "re-run as: $0 --enable-apis"
  fi
done

step "the prepared image"
IMAGE="$(image_in_family)"; IMAGE_STATUS=$?
if [ "$IMAGE_STATUS" -eq 0 ]; then
  pass "image $IMAGE in family $RC_IMAGE_FAMILY"
elif [ "$IMAGE_STATUS" -eq 1 ]; then
  # Not a failure for a new user: it is simply the first thing they do.
  printf '  [--] no image in family %s yet\n      -> build it once (~45 min, mostly a 23 GB asset download):\n         bash %s/build_image.sh\n' \
    "$RC_IMAGE_FAMILY" "$HERE"
else
  # Could not ask. Saying "no image" here would send a reader off to repeat a
  # 45-minute build that was never the problem.
  fail "could not query images, so whether one exists is unknown" "$(image_reason)"
fi

step "summary"
if [ "$PROBLEMS" -eq 0 ]; then
  if [ -n "$IMAGE" ]; then
    say "ready. next: bash $HERE/provision.sh --workers 6"
  else
    say "ready. next: bash $HERE/build_image.sh   (once, ~45 min)"
  fi
  exit 0
fi
say "$PROBLEMS problem(s) above; fix those and re-run"
exit 1
