#!/usr/bin/env bash
# Copy the interface checks to the VM and run them there.
#
# The checks are a file rather than inline ssh arguments on purpose: inlining
# them means three layers of quoting (shell, ssh, python), and the failure mode
# is a check that silently produces no output, which is worse than no check.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/common.sh"
require_project
vm_exists "$RC_VM" || die "$RC_VM does not exist; run provision.sh"

step "verifying the environment on $RC_VM"
gcloud compute scp "$HERE/checks.sh" "$RC_VM":~/checks.sh --zone="$RC_ZONE" --quiet
on_vm "bash ~/checks.sh"
