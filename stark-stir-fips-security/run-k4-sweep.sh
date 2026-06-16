#!/bin/bash
# Sequential k=4 STIR + binary-arity bench sweep, six cells of paper Table 5.
# Each run takes ~10 min on Apple Silicon; total ~80 min including 4 recompiles.

set -e
cd "/Volumes/SAHexternal 1/Documents/stark-stir-fips/stark-stir-fips-security"
OUTDIR="/Volumes/SAHexternal/Downloads/k4-bcikscurve-bench"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/sweep.log"

echo "=== k=4 STIR sweep started at $(date) ===" | tee -a "$LOG"

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

# Cell 1: L1 q=2^40 SHA3-256 FP6 r=54
run_cell "L1-SHA256-r54-q40" 54 \
  "Final-NIST-L1-k4arity-SHA256-FP6-r54-q2-40-benchmarkdata.csv" \
  "parallel,sha3-256"

# Cells 2,3: L1 q=2^65 (r=54) and L3 q=2^40 (r=79) both SHA3-384 FP6
run_cell "L1-SHA384-r54-q65" 54 \
  "Final-NIST-L1-k4arity-SHA3384-FP6-r54-q2-65-benchmarkdata.csv" \
  "parallel,sha3-384"

run_cell "L3-SHA384-r79-q40" 79 \
  "Final-NIST-L3-k4arity-SHA384-FP6-r79-q2-40-benchmarkdata.csv" \
  "parallel,sha3-384"

# Cells 4,5: L1 q=2^90 (r=54) and L3 q=2^90 (r=79) both SHA3-512 FP6
run_cell "L1-SHA512-r54-q90" 54 \
  "Final-NIST-L1-k4arity-SHA3512-FP6-r54-q2-90-benchmarkdata.csv" \
  "parallel,sha3-512"

run_cell "L3-SHA512-r79-q90" 79 \
  "Final-NIST-L3-k4arity-SHA512-FP6-r79-q2-90-q2-65-benchmarkdata.csv" \
  "parallel,sha3-512"

# Cell 6: L5 q=2^40 SHA3-512 FP8 r=105
run_cell "L5-SHA512-r105-q40" 105 \
  "Final-NIST-L5-k4arity-SHA512-FP8-r105-multiair-benchmarkdata.csv" \
  "parallel,sha3-512,ext-octic"

echo | tee -a "$LOG"
echo "=== All six cells complete at $(date) ===" | tee -a "$LOG"
ls -la "$OUTDIR" | tee -a "$LOG"
