#!/usr/bin/env bash
# regenerate-paper-tables.sh — one-shot driver: run all paper-relevant
# benches, then emit LaTeX `tabular` fragments from the resulting CSVs.
#
# After this completes, copy the three .tex fragments into the paper's
# source tree (or `\input` them directly if the paper repo can see this
# directory).
#
# Outputs:
#   scripts/results/paper_tex/paper-table-crypto-airs.tex     (Table 5)
#   scripts/results/paper_tex/paper-table-rollup.tex          (Table 7)
#   scripts/results/paper_tex/paper-table-rollup-recursive.tex (Table 8)
#
# Usage:
#   ./scripts/regenerate-paper-tables.sh                   # full re-bench + emit
#   SKIP_BENCH=1 ./scripts/regenerate-paper-tables.sh      # emit from existing CSVs
#
# Environment passthrough (see individual bench scripts for details):
#   BENCH_BLOWUP, BENCH_LDT_ONLY, BENCH_NS, RAYON_NUM_THREADS

set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_BENCH="${SKIP_BENCH:-0}"

if [[ "$SKIP_BENCH" != "1" ]]; then
    echo "═══ Re-bench: signatures (Table 5) ═══"
    ./scripts/bench-all-signatures.sh

    echo
    echo "═══ Re-bench: ML-DSA HashRollup (Table 7) ═══"
    ./scripts/bench-ml-dsa-rollup.sh

    # Recursive-rollup CSV is not yet produced by an existing script.
    # When that bench exists, drop the invocation here, e.g.:
    #   ./scripts/bench-ml-dsa-recursive-rollup.sh
    if [[ -x ./scripts/bench-ml-dsa-recursive-rollup.sh ]]; then
        echo
        echo "═══ Re-bench: ML-DSA recursive rollup (Table 8) ═══"
        ./scripts/bench-ml-dsa-recursive-rollup.sh
    else
        echo
        echo "[skip] scripts/bench-ml-dsa-recursive-rollup.sh not present;"
        echo "       Table 8 fragment will be skipped (or use existing CSV)."
    fi
fi

echo
echo "═══ Emit LaTeX fragments ═══"
python3 ./scripts/csv-to-paper-tex.py --results-dir ./scripts/results

echo
echo "Done.  Fragments at:"
echo "  ./scripts/results/paper_tex/paper-table-crypto-airs.tex"
echo "  ./scripts/results/paper_tex/paper-table-rollup.tex"
echo "  ./scripts/results/paper_tex/paper-table-rollup-recursive.tex"
