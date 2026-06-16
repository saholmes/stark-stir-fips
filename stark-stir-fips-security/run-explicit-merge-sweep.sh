#!/bin/bash
# P5.9 — Explicit-form prover/verifier sweep.
#
# Drives the EXPLICIT merge construction (deep_fri_prove_explicit +
# deep_fri_verify_explicit) at multiple NIST levels and produces a
# CSV of (prove_ms, verify_ms, proof_bytes) per cell.
#
# Companion to run-stir-halve-sweep.sh (which drives the IMPLICIT
# form's stir_halve::tests::canonical_k22_proof_size).
#
# Usage:
#   ./run-explicit-merge-sweep.sh                 # default cells
#   K_LOG=14 ./run-explicit-merge-sweep.sh        # override n = 2^K_LOG
#   R=40 ./run-explicit-merge-sweep.sh            # override query count
#
# Each cell rebuilds with a different sha3-N feature flag so the
# NIST-level cap propagates through every Merkle / FS hash.

set -e
cd "/Volumes/SAHexternal 1/Documents/stark-stir-fips/stark-stir-fips-security"

OUTDIR="/Volumes/SAHexternal/Downloads/explicit-merge-bench"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/sweep.log"
CSV="$OUTDIR/explicit-merge-benchmarkdata.csv"

# Defaults (overridable via env).
K_LOG="${K_LOG:-10}"
BLOWUP="${BLOWUP:-4}"
R="${R:-30}"

# Wipe prior CSV for reproducibility; write header.
rm -f "$CSV"
echo "label,k_log,n,trace_len,blowup,r,prove_ms,verify_ms,proof_bytes" > "$CSV"

echo "=== Explicit-form sweep (k_log=$K_LOG, blowup=$BLOWUP, r=$R) ===" | tee "$LOG"
echo "    output CSV : $CSV" | tee -a "$LOG"
echo "    log file   : $LOG"  | tee -a "$LOG"

run_cell() {
  local label="$1"; shift
  local features="$1"; shift
  echo | tee -a "$LOG"
  echo "[$label] features=$features" | tee -a "$LOG"
  echo "[$label] started $(date)" | tee -a "$LOG"
  EXPLICIT_K_LOG="$K_LOG" \
    EXPLICIT_BLOWUP="$BLOWUP" \
    EXPLICIT_R="$R" \
    EXPLICIT_LABEL="$label" \
    EXPLICIT_CSV_APPEND="$CSV" \
    cargo test -p deep_ali --release \
      --features "$features" --lib \
      explicit_merge_prove::tests::explicit_form_bench_one_cell \
      -- --ignored --nocapture 2>&1 | tail -12 | tee -a "$LOG"
  echo "[$label] finished $(date)" | tee -a "$LOG"
}

# L1: SHA-3-256 — NIST Level 1
run_cell "L1-SHA3-256" "parallel,sha3-256"

# L3: SHA-3-384 — NIST Level 3
run_cell "L3-SHA3-384" "parallel,sha3-384"

# L5: SHA-3-512 — NIST Level 5
run_cell "L5-SHA3-512" "parallel,sha3-512"

echo | tee -a "$LOG"
echo "=== Sweep complete at $(date) ===" | tee -a "$LOG"
echo "[CSV] $CSV" | tee -a "$LOG"
echo | tee -a "$LOG"
echo "csv contents:" | tee -a "$LOG"
cat "$CSV" | tee -a "$LOG"
