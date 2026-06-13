#!/bin/bash
# ÷2 STIR sweep: six paper Table 5 cells, measured against the GENUINE
# domain-halving construction (option A from the reviewer audit).  This
# is the prove+verify+proof-bytes measurement that the paper's theory
# (§2-§5) actually describes — distinct from the ÷k FRI-domain numbers
# in run-k4-sweep*.sh which are now a baseline rather than the headline.
#
# Each cell rebuilds (3 distinct SHA-3 feature flags), sets the per-NIST-
# level {t_i} schedule from paper Table 2, and runs the parameterised
# canonical test.  Per-cell wall-time at k=22, M=8: ~3s test + ~10s build
# when features change ≈ 60-90s total for the whole sweep.

set -e
cd "/Volumes/SAHexternal 1/Documents/stark-stir-fips/stark-stir-fips-security"

OUTDIR="/Volumes/SAHexternal/Downloads/stir-halve-bench"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/sweep.log"
CSV="$OUTDIR/stir-halve-benchmarkdata.csv"

# Allow override of n0 = 2^K for quicker dev runs (default k=22).
K_LOG="${K_LOG:-22}"

# Start a fresh CSV (the test appends; we wipe any prior contents so the
# sweep is reproducible end-to-end).
rm -f "$CSV"

echo "=== ÷2 STIR sweep (k_log=$K_LOG) started at $(date) ===" | tee "$LOG"
echo "    output CSV : $CSV" | tee -a "$LOG"
echo "    log file   : $LOG"  | tee -a "$LOG"

run_cell() {
  local label="$1"; shift
  local schedule="$1"; shift
  local features="$1"; shift
  local m
  m=$(echo "$schedule" | tr ',' '\n' | wc -l | tr -d ' ')
  echo | tee -a "$LOG"
  echo "[$label] schedule=$schedule features=$features M=$m" | tee -a "$LOG"
  echo "[$label] started $(date)" | tee -a "$LOG"
  STIRHALVE_K="$K_LOG" \
    STIRHALVE_M="$m" \
    STIRHALVE_T_SCHEDULE="$schedule" \
    STIRHALVE_RATE_INV=32 \
    STIRHALVE_LABEL="$label" \
    STIRHALVE_CSV_APPEND="$CSV" \
    cargo test -p deep_ali --release \
      --features "$features" --lib \
      stir_halve::tests::canonical_k22_proof_size \
      -- --ignored --nocapture 2>&1 | tail -10 | tee -a "$LOG"
  echo "[$label] finished $(date)" | tee -a "$LOG"
}

# Paper Table 2 (canonical M=8 schedules).
L1_SCHED="55,46,39,34,30,27,25,23"   # Σ = 279
L3_SCHED="81,68,58,50,45,40,37,34"   # Σ = 413
L5_SCHED="108,89,76,67,59,53,48,44"  # Σ = 544

# Cell 1: L1 SHA-3-256
run_cell "L1-SHA3-256" "$L1_SCHED" "parallel,sha3-256"

# Cell 2: L1 SHA-3-384
run_cell "L1-SHA3-384" "$L1_SCHED" "parallel,sha3-384"

# Cell 3: L3 SHA-3-384
run_cell "L3-SHA3-384" "$L3_SCHED" "parallel,sha3-384"

# Cell 4: L1 SHA-3-512
run_cell "L1-SHA3-512" "$L1_SCHED" "parallel,sha3-512"

# Cell 5: L3 SHA-3-512
run_cell "L3-SHA3-512" "$L3_SCHED" "parallel,sha3-512"

# Cell 6: L5 SHA-3-512
run_cell "L5-SHA3-512" "$L5_SCHED" "parallel,sha3-512"

echo | tee -a "$LOG"
echo "=== ÷2 STIR sweep complete at $(date) ===" | tee -a "$LOG"
echo "[CSV] $CSV"   | tee -a "$LOG"
echo                | tee -a "$LOG"
echo "csv contents:" | tee -a "$LOG"
cat "$CSV" | tee -a "$LOG"
