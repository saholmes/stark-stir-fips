#!/bin/bash
# ÷2 STIR Ext-sidecar sweep: same 6 canonical cells as run-stir-halve-sweep.sh,
# but each cell additionally builds + verifies the Fp^e (e=6 for L1/L3,
# e=8 for L5) OOD-quotient sidecar.  Reports incremental sidecar bytes,
# prove-time delta, verify-time delta — these are the deltas the paper
# Table 5 footnote describes when the Ext lift is enabled.

set -e
cd "/Volumes/SAHexternal 1/Documents/stark-stir-fips/stark-stir-fips-security"

OUTDIR="/Volumes/SAHexternal/Downloads/stir-halve-bench"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/ext-sweep.log"
CSV="$OUTDIR/stir-halve-ext-benchmarkdata.csv"

K_LOG="${K_LOG:-22}"

rm -f "$CSV"

echo "=== ÷2 STIR Ext-sidecar sweep (k_log=$K_LOG) started at $(date) ===" | tee "$LOG"
echo "    output CSV : $CSV" | tee -a "$LOG"
echo "    log file   : $LOG"  | tee -a "$LOG"

# Run one cell: $1 label, $2 schedule, $3 features, $4 test name (G6 or G8).
run_cell() {
  local label="$1"; shift
  local schedule="$1"; shift
  local features="$1"; shift
  local testname="$1"; shift
  local m
  m=$(echo "$schedule" | tr ',' '\n' | wc -l | tr -d ' ')
  echo | tee -a "$LOG"
  echo "[$label] schedule=$schedule features=$features test=$testname M=$m" | tee -a "$LOG"
  echo "[$label] started $(date)" | tee -a "$LOG"
  STIRHALVE_K="$K_LOG" \
    STIRHALVE_M="$m" \
    STIRHALVE_T_SCHEDULE="$schedule" \
    STIRHALVE_RATE_INV=32 \
    STIRHALVE_LABEL="$label" \
    STIRHALVE_CSV_APPEND="$CSV" \
    cargo test -p deep_ali --release \
      --features "$features" --lib \
      "stir_halve::tests::canonical_k22_ext_sidecar_${testname}" \
      -- --ignored --nocapture 2>&1 | tail -12 | tee -a "$LOG"
  echo "[$label] finished $(date)" | tee -a "$LOG"
}

L1_SCHED="55,46,39,34,30,27,25,23"   # Σ = 279
L3_SCHED="81,68,58,50,45,40,37,34"   # Σ = 413
L5_SCHED="108,89,76,67,59,53,48,44"  # Σ = 544

# Cell 1: L1 SHA-3-256, G6
run_cell "L1-SHA3-256" "$L1_SCHED" "parallel,sha3-256" g6
# Cell 2: L1 SHA-3-384, G6
run_cell "L1-SHA3-384" "$L1_SCHED" "parallel,sha3-384" g6
# Cell 3: L3 SHA-3-384, G6
run_cell "L3-SHA3-384" "$L3_SCHED" "parallel,sha3-384" g6
# Cell 4: L1 SHA-3-512, G6
run_cell "L1-SHA3-512" "$L1_SCHED" "parallel,sha3-512" g6
# Cell 5: L3 SHA-3-512, G6
run_cell "L3-SHA3-512" "$L3_SCHED" "parallel,sha3-512" g6
# Cell 6: L5 SHA-3-512, G8
run_cell "L5-SHA3-512" "$L5_SCHED" "parallel,sha3-512" g8

echo | tee -a "$LOG"
echo "=== ÷2 STIR Ext-sidecar sweep complete at $(date) ===" | tee -a "$LOG"
echo "[CSV] $CSV"   | tee -a "$LOG"
echo                | tee -a "$LOG"
echo "csv contents:" | tee -a "$LOG"
cat "$CSV" | tee -a "$LOG"
