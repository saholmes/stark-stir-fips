#!/usr/bin/env bash
# sweep-arity-blowup.sh — sweep folding arity k × blowup b for FRI and
# STIR across the in-tree per-signature AIRs.  Emits one CSV row per
# cell to scripts/results/sweep.csv with the columns:
#
#   phase, air, ldt, k, b, r,
#   prove_ms, verify_ms, proof_kib, n_trace, threads
#
# Soundness invariant: each cell is calibrated to NIST L1 (λ=128) via
#   - FRI uniform-r:   r = ⌈λ / log₂(b)⌉              (capacity-regime proxy)
#   - STIR uniform-r:  r = ⌈2λ / log₂(b)⌉             (Johnson half-bit floor)
# The closed-form check matches `deep_ali::sweep_calibration` helpers.
#
# Phases:
#   SWEEP_PHASE=1   small AIRs (Fibonacci, ECDSA-K=2, ML-DSA-44 b≤16) — fast iteration
#   SWEEP_PHASE=2   RSA-2048 + ML-DSA-65 at predicted optima ± neighbours
#   SWEEP_PHASE=3   Ed25519 K=256 + ML-DSA-87 (overnight cells)
#
# Usage:
#   ./scripts/sweep-arity-blowup.sh                    # Phase 1 by default
#   SWEEP_PHASE=2 ./scripts/sweep-arity-blowup.sh
#   SWEEP_PHASE=all SWEEP_DRY_RUN=1 ./scripts/sweep-arity-blowup.sh

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

RESULTS_DIR="$REPO_ROOT/scripts/results"
mkdir -p "$RESULTS_DIR"

SWEEP_PHASE="${SWEEP_PHASE:-1}"
SWEEP_DRY_RUN="${SWEEP_DRY_RUN:-0}"
LAMBDA="${SWEEP_LAMBDA:-128}"   # NIST L1 target
CSV="$RESULTS_DIR/sweep.csv"

# Header (overwrite per run for clean state)
echo "phase,air,ldt,k,b,r,prove_ms,verify_ms,proof_kib,n_trace,threads,note" > "$CSV"

NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$NPROC}"

# ── Soundness-calibration helpers (closed-form) ──────────────────────
# log2 of a power-of-two integer
log2 () { local n=$1 r=0; while [ $n -gt 1 ]; do n=$((n >> 1)); r=$((r + 1)); done; echo $r; }

# ceil(num/denom) for positive integers
ceil_div () { echo $(( ($1 + $2 - 1) / $2 )); }

# FRI uniform-r: r = ⌈λ / log₂(b)⌉
calibrate_fri () {
    local lambda=$1 blowup=$2
    local log_b; log_b=$(log2 "$blowup")
    [ "$log_b" -gt 0 ] || { echo "blowup must be > 1" >&2; exit 1; }
    ceil_div "$lambda" "$log_b"
}

# STIR uniform-r: r = ⌈2λ / log₂(b)⌉
calibrate_stir_uniform () {
    local lambda=$1 blowup=$2
    local log_b; log_b=$(log2 "$blowup")
    [ "$log_b" -gt 0 ] || { echo "blowup must be > 1" >&2; exit 1; }
    ceil_div "$((2 * lambda))" "$log_b"
}

# ── Cell skip rules ──────────────────────────────────────────────────
# Skip when:
#   - n_0 not a power of k (schedule arithmetic ill-formed; bench will
#     assert and abort, which we want to avoid pre-flight)
#   - LDE memory likely OOM (k≥8 with large AIRs at b≥32)
skip_cell () {
    local air=$1 ldt=$2 k=$3 b=$4
    case "$air" in
        # Ed25519 K=256: huge trace; skip k≥8 at any blowup.
        ed25519)
            [ "$k" -ge 8 ] && return 0
            [ "$b" -le 8 ] && return 0   # under-secured at small blowup
            ;;
        # ECDSA-p256 K=2 stub: tiny trace; skip k=16 + b=64
        ecdsa)
            [ "$k" -eq 16 ] && [ "$b" -ge 32 ] && return 0
            ;;
        # ML-DSA composite at b≥32 + k≥8 → ~50 MiB proofs, multi-hour
        mldsa_v2)
            [ "$k" -ge 8 ] && [ "$b" -ge 32 ] && return 0
            ;;
        # RSA-2048: cap k=8 at b=32 for tractability; k=16 only at b≤16
        rsa2048)
            [ "$k" -ge 16 ] && [ "$b" -ge 32 ] && return 0
            ;;
    esac
    return 1
}

# Phase membership
in_phase () {
    local air=$1 b=$2 phase=$3
    case "$phase" in
        1)
            case "$air" in
                ecdsa)  [ "$b" -le 16 ] && return 0 ;;
                mldsa_v2) [ "$b" -le 16 ] && return 0 ;;
                # Phase 1 = small AIRs only
            esac
            return 1
            ;;
        2)
            case "$air" in
                rsa2048) [ "$b" -ge 16 ] && [ "$b" -le 32 ] && return 0 ;;
                mldsa_v2) [ "$b" -eq 32 ] && return 0 ;;
            esac
            return 1
            ;;
        3)
            case "$air" in
                ed25519) return 0 ;;       # all cells (skipping handled above)
                mldsa_v2) [ "$b" -ge 32 ] && return 0 ;;
            esac
            return 1
            ;;
        all) return 0 ;;
    esac
    return 1
}

# ── Per-AIR runners ──────────────────────────────────────────────────
run_cell () {
    local air=$1 ldt=$2 k=$3 b=$4 r=$5
    local features prove_ms verify_ms proof_kib n_trace note=""
    local log="$RESULTS_DIR/sweep-${air}-${ldt}-k${k}-b${b}.log"

    # Pre-flight: assert n_0 alignment via cheap dry-run check
    # (we let the bench itself enforce; just log the intent here).

    case "$air" in
        rsa2048)
            features="parallel sha3-256 mldsa-44"
            export BENCH_LDT="$ldt" BENCH_BLOWUP="$b" BENCH_QUERIES="$r" BENCH_FOLD_K="$k"
            unset BENCH_T_SCHEDULE
            if [ "$SWEEP_DRY_RUN" = "1" ]; then
                prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note=dry-run
            else
                local line
                if line=$(cargo run --release -p deep_ali --example rsa2048_bench \
                            --features "$features" --no-default-features 2>&1 | tee "$log" \
                            | grep "^rsa2048_bench " | tail -1); then
                    prove_ms=$(echo "$line" | grep -oE "prove_ms=[0-9.]+" | cut -d= -f2)
                    verify_ms=$(echo "$line" | grep -oE "verify_ms=[0-9.]+" | cut -d= -f2)
                    proof_kib=$(echo "$line" | grep -oE "proof_kib=[0-9.]+" | cut -d= -f2)
                    n_trace=$(echo "$line"   | grep -oE "n_trace=[0-9]+"    | cut -d= -f2)
                else
                    prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note=bench-error
                fi
            fi
            ;;
        ed25519)
            features="parallel sha3-256 mldsa-44"
            export BENCH_LDT="$ldt" BENCH_BLOWUP="$b" BENCH_QUERIES="$r" BENCH_FOLD_K="$k"
            export BENCH_K_SCALAR="${BENCH_K_SCALAR:-256}"
            unset BENCH_T_SCHEDULE
            if [ "$SWEEP_DRY_RUN" = "1" ]; then
                prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note=dry-run
            else
                local line
                if line=$(cargo run --release -p deep_ali --example ed25519_bench \
                            --features "$features" --no-default-features 2>&1 | tee "$log" \
                            | grep "^ed25519_bench " | tail -1); then
                    prove_ms=$(echo "$line" | grep -oE "prove_ms=[0-9.]+" | cut -d= -f2)
                    verify_ms=$(echo "$line" | grep -oE "verify_ms=[0-9.]+" | cut -d= -f2)
                    proof_kib=$(echo "$line" | grep -oE "proof_kib=[0-9.]+" | cut -d= -f2)
                    n_trace=$(echo "$line"   | grep -oE "n_trace=[0-9]+"    | cut -d= -f2)
                else
                    prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note=bench-error
                fi
            fi
            ;;
        ecdsa)
            features="parallel sha3-256 mldsa-44 p256-merge-helpers"
            export BENCH_LDT="$ldt" BENCH_BLOWUP="$b" BENCH_QUERIES="$r" BENCH_FOLD_K="$k"
            export BENCH_K_SCALAR="${BENCH_K_SCALAR:-2}"
            unset BENCH_T_SCHEDULE
            if [ "$SWEEP_DRY_RUN" = "1" ]; then
                prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note=dry-run
            else
                local line
                if line=$(cargo run --release -p deep_ali --example ecdsa_p256_bench \
                            --features "$features" --no-default-features 2>&1 | tee "$log" \
                            | grep "^ecdsa_p256_bench " | tail -1); then
                    prove_ms=$(echo "$line" | grep -oE "prove_ms=[0-9.]+" | cut -d= -f2)
                    verify_ms=$(echo "$line" | grep -oE "verify_ms=[0-9.]+" | cut -d= -f2)
                    proof_kib=$(echo "$line" | grep -oE "proof_kib=[0-9.]+" | cut -d= -f2)
                    n_trace=$(echo "$line"   | grep -oE "n_trace=[0-9]+"    | cut -d= -f2)
                else
                    prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note=bench-error
                fi
            fi
            ;;
        mldsa_v2)
            features="parallel sha3-256 mldsa-44 mldsa-merge-helpers"
            export BENCH_BLOWUP="$b" BENCH_FOLD_K="$k"
            case "$ldt" in
                stir) unset MMIYC_V2_USE_FRI ;;
                fri)  export MMIYC_V2_USE_FRI=1 ;;
            esac
            unset BENCH_T_SCHEDULE BENCH_QUERIES
            if [ "$SWEEP_DRY_RUN" = "1" ]; then
                prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note=dry-run
            else
                local line
                if line=$(cargo test --release -p deep_ali \
                            --features "$features" --no-default-features \
                            --lib v2_bench -- --ignored --nocapture 2>&1 | tee "$log" \
                            | grep "^v2_bench " | tail -1); then
                    prove_ms=$(echo "$line" | grep -oE "prove_ms=[0-9.]+" | cut -d= -f2)
                    verify_ms=$(echo "$line" | grep -oE "verify_ms=[0-9.]+" | cut -d= -f2)
                    proof_kib=$(echo "$line" | grep -oE "proof_kib=[0-9.]+" | cut -d= -f2)
                    n_trace=NA
                else
                    prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note=bench-error
                fi
            fi
            ;;
        *)
            echo "unknown AIR: $air" >&2
            exit 1
            ;;
    esac

    : "${prove_ms:=NA}"; : "${verify_ms:=NA}"; : "${proof_kib:=NA}"; : "${n_trace:=NA}"
    echo "${SWEEP_PHASE},${air},${ldt},${k},${b},${r},${prove_ms},${verify_ms},${proof_kib},${n_trace},${RAYON_NUM_THREADS},${note}" >> "$CSV"
    printf "  → %-12s ldt=%-4s k=%-2d b=%-2d r=%-4d  prove=%sms verify=%sms proof=%sKiB %s\n" \
        "$air" "$ldt" "$k" "$b" "$r" "$prove_ms" "$verify_ms" "$proof_kib" "$note"
}

# ── Sweep grid ────────────────────────────────────────────────────────
LDTS=("stir" "fri")
KS=(2 4 8 16)
BS=(4 8 16 32 64)
AIRS=("rsa2048" "ecdsa" "mldsa_v2" "ed25519")

PHASES=("$SWEEP_PHASE")
[ "$SWEEP_PHASE" = "all" ] && PHASES=(1 2 3)

echo "## Sweep started: $(date -Iseconds)"
echo "## Phase(s): ${PHASES[*]}"
echo "## λ = $LAMBDA, dry-run=$SWEEP_DRY_RUN"
echo

for phase in "${PHASES[@]}"; do
    SWEEP_PHASE="$phase"
    echo "═════ Phase $phase ═════"
    for air in "${AIRS[@]}"; do
        for ldt in "${LDTS[@]}"; do
            for k in "${KS[@]}"; do
                for b in "${BS[@]}"; do
                    if skip_cell "$air" "$ldt" "$k" "$b"; then continue; fi
                    if ! in_phase "$air" "$b" "$phase"; then continue; fi
                    case "$ldt" in
                        fri)  r=$(calibrate_fri          "$LAMBDA" "$b") ;;
                        stir) r=$(calibrate_stir_uniform "$LAMBDA" "$b") ;;
                    esac
                    run_cell "$air" "$ldt" "$k" "$b" "$r"
                done
            done
        done
    done
done

echo
echo "═══════════════════════════════════════════════════════════"
echo "Done.  CSV: $CSV"
echo "Rows: $(($(wc -l < "$CSV") - 1))"
echo "═══════════════════════════════════════════════════════════"
