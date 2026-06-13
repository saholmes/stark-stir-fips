#!/bin/bash
# ÷2 halving full-Ext sweep at BINARY FRI fold (k=2) — the |Z|=1
# instance of the same ÷2 halving construction family that the
# STIR k=4 sweep (run-stir-halve-full-ext-sweep.sh) measures.
#
# Same six canonical (level, hash) cells, same n0 = 2^22, same
# canonical {t_i} schedule.  Only the fold arity changes
# (STIRHALVE_DEG_DIV=2 instead of the default 4).  Demonstrates the
# paper's "one merge family, both LDTs in one prover" claim: the
# same prove_halve_full_ext / verify_halve_full_ext code handles
# both binary FRI and k=4 STIR by changing a single parameter.
#
# Note: the {t_i} schedule is the STIR declining schedule, used here
# for apples-to-apples bytes / wallclock comparison at the same
# query count.  A binary-FRI Johnson-regime soundness analysis
# would prefer a flat t_i schedule (rate doesn't decline at k=2).

set -e
cd "/Volumes/SAHexternal 1/Documents/stark-stir-fips/stark-stir-fips-security"

OUTDIR="/Volumes/SAHexternal/Downloads/stir-halve-bench"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/full-ext-fri-sweep.log"
CSV="$OUTDIR/stir-halve-full-ext-fri-benchmarkdata.csv"

K_LOG="${K_LOG:-22}"

rm -f "$CSV"

echo "=== ÷2 halving full-Ext BINARY-FRI sweep (k_log=$K_LOG, deg_div=2) started at $(date) ===" | tee "$LOG"

run_cell() {
  local label="$1"; shift
  local schedule="$1"; shift
  local features="$1"; shift
  local testname="$1"; shift
  local m
  m=$(echo "$schedule" | tr ',' '\n' | wc -l | tr -d ' ')
  echo | tee -a "$LOG"
  echo "[$label] schedule=$schedule features=$features test=$testname M=$m (deg_div=2)" | tee -a "$LOG"
  echo "[$label] started $(date)" | tee -a "$LOG"
  STIRHALVE_K="$K_LOG" \
    STIRHALVE_M="$m" \
    STIRHALVE_T_SCHEDULE="$schedule" \
    STIRHALVE_RATE_INV=32 \
    STIRHALVE_LABEL="$label" \
    STIRHALVE_CSV_APPEND="$CSV" \
    STIRHALVE_DEG_DIV=2 \
    cargo test -p deep_ali --release \
      --features "$features" --lib \
      "stir_halve::tests::canonical_k22_full_ext_${testname}" \
      -- --ignored --nocapture 2>&1 | tail -10 | tee -a "$LOG"
  echo "[$label] finished $(date)" | tee -a "$LOG"
}

L1_SCHED="55,46,39,34,30,27,25,23"
L3_SCHED="81,68,58,50,45,40,37,34"
L5_SCHED="108,89,76,67,59,53,48,44"

run_cell "L1-SHA3-256" "$L1_SCHED" "parallel,sha3-256" g6
run_cell "L1-SHA3-384" "$L1_SCHED" "parallel,sha3-384" g6
run_cell "L1-SHA3-512" "$L1_SCHED" "parallel,sha3-512" g6
run_cell "L3-SHA3-384" "$L3_SCHED" "parallel,sha3-384" g6
run_cell "L3-SHA3-512" "$L3_SCHED" "parallel,sha3-512" g6
run_cell "L5-SHA3-512" "$L5_SCHED" "parallel,sha3-512" g8

echo | tee -a "$LOG"
echo "=== ÷2 halving full-Ext BINARY-FRI sweep complete at $(date) ===" | tee -a "$LOG"
cat "$CSV" | tee -a "$LOG"
