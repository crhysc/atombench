#!/usr/bin/env bash
# depcheck.sh — verbose dependency checker for Linux + Slurm + CUDA module + conda hook
# Usage: bash depcheck.sh
#
# This script is intentionally chatty: it explains *exactly* what failed and how to fix it.

set -u  # treat unset vars as errors (but we won't use set -e; we want custom diagnostics)

PASS_COUNT=0
FAIL_COUNT=0

hr() { printf '%s\n' "------------------------------------------------------------"; }
ok() { printf '✅ %s\n' "$1"; PASS_COUNT=$((PASS_COUNT+1)); }
warn() { printf '⚠️  %s\n' "$1"; }
fail() { printf '❌ %s\n' "$1"; FAIL_COUNT=$((FAIL_COUNT+1)); }

section() {
  hr
  printf '%s\n' "$1"
  hr
}

# Run a command, capture output, and report with context.
run_capture() {
  # args: <label> <command...>
  local label="$1"; shift
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  printf '%s\n' "$out"
  return "$rc"
}

# ---------- 1) Linux check ----------
check_linux() {
  section "1) OS check (must be Linux)"

  local uname_s
  uname_s="$(uname -s 2>/dev/null || true)"

  if [[ "$uname_s" == "Linux" ]]; then
    ok "Detected Linux (uname -s = Linux)."
    return 0
  fi

  fail "Not Linux: uname -s returned '${uname_s:-<empty>}'.
This script expects a Linux environment (typical for HPC login nodes)."
  return 1
}

# ---------- 2) Slurm present ----------
check_slurm() {
  section "2) Slurm check (must have Slurm commands available)"

  # Prefer squeue; fallback to sbatch/sinfo
  if command -v squeue >/dev/null 2>&1; then
    ok "Found Slurm command: squeue ($(command -v squeue))."
  elif command -v sbatch >/dev/null 2>&1; then
    ok "Found Slurm command: sbatch ($(command -v sbatch))."
  elif command -v sinfo >/dev/null 2>&1; then
    ok "Found Slurm command: sinfo ($(command -v sinfo))."
  else
    fail "Slurm does not appear to be present in PATH.
Tried: squeue, sbatch, sinfo.
Fix: load your site environment/modules, or use the correct login node, or add Slurm bin to PATH."
    return 1
  fi

  # Optional: sanity-call squeue (doesn't require allocations, but may fail if Slurm is down)
  if command -v squeue >/dev/null 2>&1; then
    local out rc
    out="$(squeue -h 2>&1)"; rc=$?
    if [[ $rc -eq 0 ]]; then
      ok "squeue runs successfully."
    else
      warn "squeue exists but returned a non-zero exit code ($rc).
Output:
$out
This may indicate a transient Slurm/controller issue, or a permissions/config problem."
    fi
  fi

  return 0
}

# ---------- 3) module load cuda/11.8 works ----------
check_cuda_module() {
  section "3) Environment Modules check (module load cuda/11.8 must work)"

  # Many clusters define `module` as a shell function; ensure it's available.
  if ! type module >/dev/null 2>&1; then
    warn "The 'module' command is not currently available in this shell."
    warn "Attempting to initialize Environment Modules (common paths)..."

    # Common init scripts across distros/HPC sites:
    if [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/Modules/init/bash ]]; then
      # shellcheck disable=SC1091
      source /usr/share/Modules/init/bash
    elif [[ -f /usr/share/modules/init/bash ]]; the
