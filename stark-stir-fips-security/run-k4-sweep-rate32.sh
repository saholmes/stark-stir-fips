#!/bin/bash
# Corrected k=4 STIR + binary-FRI sweep at rate 1/32 (paper-claimed
# Johnson-regime parameters).  The prior k4-bcikscurve-bench run used
# the default real_trace_inputs(n0, 4) which gives rate 1/4 — that
# under-claims the security level by a factor of 2.5 (Johnson per-query
# yield: 1.0 bit/query at rho=1/4 vs 2.5 bits/query at rho=1/32).
#
# This script:
#   1. Patches end_to_end.rs in-place: 4 -> 32 for trace_inputs + n_trace
#   2. Re-runs the 6 cells at the corrected blowup
#   3. Reverts the patch at the end (idempotent)
#
# Total wall-clock: ~80-120 min (slightly longer than the 1/4 run since
# the prover does more FFT work at the larger blowup).

set -e
cd "/Volumes/SAHexternal 1/Documents/stark-stir-fips/stark-stir-fips-security"
OUTDIR="/Volumes/SAHexternal/Downloads/k4-rate32-bench"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/sweep.log"

BENCH_FILE="crates/channel/benches/end_to_end.rs"

# ── Sanity check ──
if ! grep -q 'real_trace_inputs(n0, 4)' "$BENCH_FILE"; then
  echo "ERROR: expected 'real_trace_inputs(n0, 4)' in $BENCH_FILE" | tee -a "$LOG"
  echo "       The bench may already be patched or the harness changed." | tee -a "$LOG"
  exit 1
fi

echo "=== rate-1/32 corrected sweep started at $(date) ===" | tee -a "$LOG"

# ── Patch the harness ──
echo "[PATCH] real_trace_inputs(n0, 4) -> (n0, 32);  n_trace = n0/4 -> n0/32" | tee -a "$LOG"
sed -i.rate4bak \
  -e 's|real_trace_inputs(n0, 4)|real_trace_inputs(n0, 32)|' \
  -e 's|let n_trace = n0 / 4;|let n_trace = n0 / 32;|' \
  -e 's|degree-bounded, rate 1/4|degree-bounded, rate 1/32|' \
  -e 's|rate 1/4 ⇒ n_trace = n0/4|rate 1/32 ⇒ n_trace = n0/32|' \
  "$BENCH_FILE"

revert_patch() {
  echo "[REVERT] restoring $BENCH_FILE" | tee -a "$LOG"
  mv "$BENCH_FILE.rate4bak" "$BENCH_FILE" 2>/dev/null || true
}
trap revert_patch EXIT

run_cell() {
  local label="$1"; shift
  local r="$1"; shift
  local out="$1"; shift
  local features="$*"
  echo | tee -a "$LOG"
  echo "[$label] r=$r features=$features → $out" | tee -a "$LOG"
  echo "[$label] started $(date)" | tee -a "$LOG"
  BENCH_R="$r" BENCH_CSV_OUT="$OUTDIR/$out" \
    cargo bench --bench end_to_end -p channel \
      --no-default-features --features "$features" \
      -- --quick 2>&1 | tail -5 | tee -a "$LOG"
  echo "[$label] finished $(date)" | tee -a "$LOG"
}

# Cell 1: L1 q=2^40 SHA3-256 FP6 r=55 (matched to +1-margin schedule)
run_cell "L1-SHA256-r55-q40" 55 \
  "Final-NIST-L1-rho32-SHA256-FP6-r55-q2-40-benchmarkdata.csv" \
  "parallel,sha3-256"

# Cells 2,3: L1 q=2^65 (r=55) and L3 q=2^40 (r=81) both SHA3-384 FP6
run_cell "L1-SHA384-r55-q65" 55 \
  "Final-NIST-L1-rho32-SHA3384-FP6-r55-q2-65-benchmarkdata.csv" \
  "parallel,sha3-384"

run_cell "L3-SHA384-r81-q40" 81 \
  "Final-NIST-L3-rho32-SHA384-FP6-r81-q2-40-benchmarkdata.csv" \
  "parallel,sha3-384"

# Cells 4,5: L1 q=2^90 (r=55) and L3 q=2^90 (r=81) both SHA3-512 FP6
run_cell "L1-SHA512-r55-q90" 55 \
  "Final-NIST-L1-rho32-SHA3512-FP6-r55-q2-90-benchmarkdata.csv" \
  "parallel,sha3-512"

run_cell "L3-SHA512-r81-q90" 81 \
  "Final-NIST-L3-rho32-SHA512-FP6-r81-q2-90-q2-65-benchmarkdata.csv" \
  "parallel,sha3-512"

# Cell 6: L5 q=2^40 SHA3-512 FP8 r=108 (matched to +1-margin schedule)
run_cell "L5-SHA512-r108-q40" 108 \
  "Final-NIST-L5-rho32-SHA512-FP8-r108-benchmarkdata.csv" \
  "parallel,sha3-512,ext-octic"

echo | tee -a "$LOG"
echo "=== All six rho=1/32 cells complete at $(date) ===" | tee -a "$LOG"
ls -la "$OUTDIR" | tee -a "$LOG"
