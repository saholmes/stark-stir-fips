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
# NIST level sweep: comma-separated subset of {1,3,5}; default = all 3.
SWEEP_NIST="${SWEEP_NIST:-1,3,5}"
# Quantum-query budget log_2 values to sweep; default covers below /
# above the L1/L3/L5 binding walls.
SWEEP_Q_LOGS="${SWEEP_Q_LOGS:-40,65,90}"
CSV="$RESULTS_DIR/sweep.csv"

# Header: write only if missing so re-runs accumulate rather than wipe.
# Use SWEEP_RESET=1 to force a clean header.  Extended header includes
# (nist_level, lambda, q_log, hash, ext_e) for the L1/L3/L5 × q sweep.
EXPECTED_HEADER="phase,nist,lambda,q_log,hash,ext_e,air,ldt,k,b,r,prove_ms,verify_ms,proof_kib,n_trace,threads,note"
if [ "${SWEEP_RESET:-0}" = "1" ] || [ ! -f "$CSV" ] \
   || [ "$(head -1 "$CSV" 2>/dev/null)" != "$EXPECTED_HEADER" ]; then
    echo "$EXPECTED_HEADER" > "$CSV"
fi

NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$NPROC}"

# ── Soundness-calibration helpers (closed-form) ──────────────────────
# log2 of a power-of-two integer
log2 () { local n=$1 r=0; while [ $n -gt 1 ]; do n=$((n >> 1)); r=$((r + 1)); done; echo $r; }

# ceil(num/denom) for positive integers
ceil_div () { echo $(( ($1 + $2 - 1) / $2 )); }

# ── Cell-tuple grid per NIST λ-target ────────────────────────────────
# Each cell-tuple is "hash:mldsa:ext_tower:q_log" where ext_tower is
# 'hex' (F_p^6, no tower-octic) or 'oct' (F_p^8, tower-octic).  The
# hash function is an independent grid axis (not derived from λ): we
# pick the SHA-3 variant whose κ_bind ceiling clears λ at the given q.
#
# Per user grid (2026-06-18):
#   L1 (λ=128): {sha3-256 @ q=2^40, sha3-384 @ q=2^65, sha3-512 @ q=2^90}
#   L3 (λ=192): {sha3-384 @ q=2^40, sha3-384 @ q=2^65 (at-wall),
#                sha3-512 @ q=2^90}
#   L5 (λ=256): {sha3-512 @ q=2^40, sha3-512 @ q=2^65}
#               (q=2^90 excluded — breaches SHA3-512 capacity at L5)
# Extension field tracks λ-target: F_p^6 at L1/L3, F_p^8 at L5
# (the orchestration's Ext alias gates on tower-octic; binding-stack
#  Ext alias gates on sha3-512).
cells_for_lambda () {
    case "$1" in
        128)  # L1
            echo "sha3-256:mldsa-44:hex:40 sha3-384:mldsa-44:hex:65 sha3-512:mldsa-44:hex:90" ;;
        192)  # L3
            echo "sha3-384:mldsa-65:hex:40 sha3-384:mldsa-65:hex:65 sha3-512:mldsa-65:hex:90" ;;
        256)  # L5
            echo "sha3-512:mldsa-87:oct:40 sha3-512:mldsa-87:oct:65" ;;
        *) echo "" ;;
    esac
}

# Hash feature → output bits.
hash_bits_for () {
    case "$1" in
        sha3-256) echo 256 ;;
        sha3-384) echo 384 ;;
        sha3-512) echo 512 ;;
        *) echo 0 ;;
    esac
}

# Tower spec → (ext_e, extra_features).
ext_for_tower () {
    case "$1" in
        hex) echo "6 ''" ;;            # F_p^6, no extra features
        oct) echo "8 tower-octic" ;;   # F_p^8 via tower-octic
        *) echo "" ;;
    esac
}

# λ → nist level (informational, for CSV/log).
nist_level_for_lambda () {
    case "$1" in
        128) echo 1 ;;
        192) echo 3 ;;
        256) echo 5 ;;
        *) echo 0 ;;
    esac
}

# ── κ-ceiling helpers (match sweep_calibration::kappa_{bind,fs}) ─────
# kappa_bind(n_hash_bits, q_log, M) = n - 3·q_log - ceil(log_2(M+2))
kappa_bind () {
    local n=$1 q=$2 m=$3
    # ceil(log_2(M+2)): inline approximation (M+2 ≤ 16 → ceil ∈ {1..4})
    local m_plus_2=$((m + 2))
    local m_log; m_log=$(log2 $(( 1 << ( $(log2 $m_plus_2) + (m_plus_2 != (1 << $(log2 $m_plus_2)) ) ) )))
    # Simpler: ceil(log_2(m+2)) via power-of-two round up
    local pow=1 lc=0
    while [ $pow -lt $m_plus_2 ]; do pow=$((pow * 2)); lc=$((lc + 1)); done
    echo $(( n - 3*q - lc ))
}

# kappa_fs(e, q_log, K) = 64·e - 2·q_log - 2·ceil(log_2(K))
kappa_fs () {
    local e=$1 q=$2 k=$3
    # ceil(log_2(K))
    local pow=1 lc=0
    while [ $pow -lt $k ]; do pow=$((pow * 2)); lc=$((lc + 1)); done
    [ $lc -lt 1 ] && lc=1
    echo $(( 64*e - 2*q - 2*lc ))
}

# wall_check(lambda, n, e, q_log, M) → "ok" or "wall:reason"
wall_check () {
    local lam=$1 n=$2 e=$3 q=$4 m=$5
    local kb; kb=$(kappa_bind "$n" "$q" "$m")
    if [ "$kb" -lt "$lam" ]; then
        echo "wall:bind=${kb}<lambda=${lam}"
        return
    fi
    local kfs; kfs=$(kappa_fs "$e" "$q" "$((m + 2))")
    if [ "$kfs" -lt "$lam" ]; then
        echo "wall:fs=${kfs}<lambda=${lam}"
        return
    fi
    echo "ok"
}

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

# ── Empirical n_trace per AIR (used for k-alignment pre-flight) ──────
# These match the bench examples' fixed n_trace values; if the bench
# changes its n_trace they should be updated here too.
n_trace_log2_for () {
    case "$1" in
        ecdsa)     echo 3 ;;   # K=2 stub: n_trace = 8 = 2^3
        rsa2048)   echo 12 ;;  # n_trace = 4096 = 2^12
        ed25519)   echo 10 ;;  # K=256 full scalar: n_trace = 1024 = 2^10
        mldsa_v2)  echo 13 ;;  # smallest sub-AIR n_trace = 8192 = 2^13
        *) echo 0; return 1 ;;
    esac
}

# Is (n_trace, k, b) k-aligned?  log_2(n_0) must be divisible by log_2(k).
is_k_aligned () {
    local air=$1 k=$2 b=$3
    local lnt; lnt=$(n_trace_log2_for "$air") || return 1
    local lb; lb=$(log2 "$b")
    local ln0=$((lnt + lb))
    local lk; lk=$(log2 "$k")
    [ "$lk" -gt 0 ] || return 1
    [ $((ln0 % lk)) -eq 0 ]
}

# Already in CSV at this (nist, q, air, ldt, k, b)?  Cell skipped if so.
already_run () {
    local nist=$1 q=$2 air=$3 ldt=$4 k=$5 b=$6
    # CSV layout: phase,nist,lambda,q_log,hash,ext_e,air,ldt,k,b,r,...
    grep -qE "^[0-9]+,${nist},[0-9]+,${q},[^,]*,[0-9]+,${air},${ldt},${k},${b}," \
        "$CSV" 2>/dev/null
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

# ── Build per-AIR feature flags for a given NIST level ───────────────
features_for () {
    local air=$1 hash=$2 mldsa=$3 extra=$4
    local base="parallel $hash $mldsa"
    [ -n "$extra" ] && [ "$extra" != "''" ] && base="$base $extra"
    case "$air" in
        rsa2048|ed25519) echo "$base" ;;
        ecdsa)           echo "$base p256-merge-helpers" ;;
        mldsa_v2)        echo "$base mldsa-merge-helpers" ;;
        *) echo "" ;;
    esac
}

# ── Per-AIR runners (NIST-level aware) ───────────────────────────────
run_cell () {
    local air=$1 ldt=$2 k=$3 b=$4 r=$5
    local nist=$6 lambda=$7 q_log=$8 hash=$9 ext_e=${10}
    local mldsa=${11} extra=${12} cell_note=${13:-}
    local features prove_ms verify_ms proof_kib n_trace note="$cell_note"
    local log="$RESULTS_DIR/sweep-${air}-${ldt}-k${k}-b${b}-L${nist}-${hash}-q${q_log}.log"

    features=$(features_for "$air" "$hash" "$mldsa" "$extra")

    case "$air" in
        rsa2048)
            export BENCH_LDT="$ldt" BENCH_BLOWUP="$b" BENCH_QUERIES="$r" BENCH_FOLD_K="$k"
            unset BENCH_T_SCHEDULE
            if [ "$SWEEP_DRY_RUN" = "1" ]; then
                prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note="${cell_note:+${cell_note};}dry-run"
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
                    prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note="${cell_note:+${cell_note};}bench-error"
                fi
            fi
            ;;
        ed25519)
            export BENCH_LDT="$ldt" BENCH_BLOWUP="$b" BENCH_QUERIES="$r" BENCH_FOLD_K="$k"
            export BENCH_K_SCALAR="${BENCH_K_SCALAR:-256}"
            unset BENCH_T_SCHEDULE
            if [ "$SWEEP_DRY_RUN" = "1" ]; then
                prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note="${cell_note:+${cell_note};}dry-run"
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
                    prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note="${cell_note:+${cell_note};}bench-error"
                fi
            fi
            ;;
        ecdsa)
            export BENCH_LDT="$ldt" BENCH_BLOWUP="$b" BENCH_QUERIES="$r" BENCH_FOLD_K="$k"
            export BENCH_K_SCALAR="${BENCH_K_SCALAR:-2}"
            unset BENCH_T_SCHEDULE
            if [ "$SWEEP_DRY_RUN" = "1" ]; then
                prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note="${cell_note:+${cell_note};}dry-run"
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
                    prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note="${cell_note:+${cell_note};}bench-error"
                fi
            fi
            ;;
        mldsa_v2)
            export BENCH_BLOWUP="$b" BENCH_FOLD_K="$k"
            case "$ldt" in
                stir) unset MMIYC_V2_USE_FRI ;;
                fri)  export MMIYC_V2_USE_FRI=1 ;;
            esac
            unset BENCH_T_SCHEDULE BENCH_QUERIES
            if [ "$SWEEP_DRY_RUN" = "1" ]; then
                prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note="${cell_note:+${cell_note};}dry-run"
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
                    prove_ms=NA; verify_ms=NA; proof_kib=NA; n_trace=NA; note="${cell_note:+${cell_note};}bench-error"
                fi
            fi
            ;;
        *)
            echo "unknown AIR: $air" >&2
            exit 1
            ;;
    esac

    : "${prove_ms:=NA}"; : "${verify_ms:=NA}"; : "${proof_kib:=NA}"; : "${n_trace:=NA}"
    # Only write to the CSV on real runs; dry-runs are observation-only
    # to avoid polluting the canonical artifact with NA placeholder rows.
    if [ "$SWEEP_DRY_RUN" != "1" ]; then
        echo "${SWEEP_PHASE},${nist},${lambda},${q_log},${hash},${ext_e},${air},${ldt},${k},${b},${r},${prove_ms},${verify_ms},${proof_kib},${n_trace},${RAYON_NUM_THREADS},${note}" >> "$CSV"
    fi
    printf "  → L%d q=2^%-2d %-12s ldt=%-4s k=%-2d b=%-2d r=%-4d  prove=%sms verify=%sms proof=%sKiB %s\n" \
        "$nist" "$q_log" "$air" "$ldt" "$k" "$b" "$r" "$prove_ms" "$verify_ms" "$proof_kib" "$note"
}

# Wall-breach row: record without invoking the bench.
record_wall_breach () {
    local air=$1 ldt=$2 k=$3 b=$4 r=$5
    local nist=$6 lambda=$7 q_log=$8 hash=$9 ext_e=${10} reason=${11}
    echo "${SWEEP_PHASE},${nist},${lambda},${q_log},${hash},${ext_e},${air},${ldt},${k},${b},${r},NA,NA,NA,NA,${RAYON_NUM_THREADS},${reason}" >> "$CSV"
    printf "  ⊘ L%d q=2^%-2d %-12s ldt=%-4s k=%-2d b=%-2d  %s\n" \
        "$nist" "$q_log" "$air" "$ldt" "$k" "$b" "$reason"
}

# ── Sweep grid ────────────────────────────────────────────────────────
LDTS=("stir" "fri")
KS=(2 4 8 16)
BS=(4 8 16 32 64)
AIRS=("rsa2048" "ecdsa" "mldsa_v2" "ed25519")

PHASES=("$SWEEP_PHASE")
[ "$SWEEP_PHASE" = "all" ] && PHASES=(1 2 3)

# Parse λ-targets subset (default: all three NIST levels).
SWEEP_LAMBDAS="${SWEEP_LAMBDAS:-128,192,256}"
IFS=',' read -ra LAMBDA_TARGETS <<< "$SWEEP_LAMBDAS"

echo "## Sweep started: $(date -Iseconds)"
echo "## Phase(s): ${PHASES[*]}"
echo "## λ-targets: ${LAMBDA_TARGETS[*]}"
echo "## dry-run=$SWEEP_DRY_RUN"
echo

for phase in "${PHASES[@]}"; do
    SWEEP_PHASE="$phase"
    echo "═════ Phase $phase ═════"
    for lambda in "${LAMBDA_TARGETS[@]}"; do
        nist=$(nist_level_for_lambda "$lambda")
        cells=$(cells_for_lambda "$lambda")
        if [ -z "$cells" ]; then
            echo "skip: no cells for λ=$lambda" >&2; continue
        fi
        echo "──── L${nist} (λ=${lambda}) ────"
        for cell_tuple in $cells; do
            IFS=':' read -r hash mldsa tower q_log <<< "$cell_tuple"
            n_hash=$(hash_bits_for "$hash")
            read -r ext_e extra <<< "$(ext_for_tower "$tower")"
            # Post-hoc κ_bind / κ_fs at this (n, e, q, M=8).
            kbind=$(kappa_bind "$n_hash" "$q_log" 8)
            kfs=$(kappa_fs "$ext_e" "$q_log" 10)
            cell_note=""
            if [ "$kbind" -lt "$lambda" ]; then
                cell_note="at-wall(bind=${kbind}<${lambda})"
            fi
            echo "── (${hash}, ${mldsa}, F_p^${ext_e}, q=2^${q_log}) κ_bind=${kbind}, κ_fs=${kfs} ${cell_note}"
            for air in "${AIRS[@]}"; do
                for ldt in "${LDTS[@]}"; do
                    for k in "${KS[@]}"; do
                        for b in "${BS[@]}"; do
                            if skip_cell "$air" "$ldt" "$k" "$b"; then continue; fi
                            if ! in_phase "$air" "$b" "$phase"; then continue; fi
                            if ! is_k_aligned "$air" "$k" "$b"; then continue; fi
                            if already_run "$nist" "$q_log" "$air" "$ldt" "$k" "$b"; then
                                printf "  ↺ L%d q=2^%-2d %-12s ldt=%-4s k=%-2d b=%-2d (already in CSV)\n" \
                                    "$nist" "$q_log" "$air" "$ldt" "$k" "$b"
                                continue
                            fi
                            case "$ldt" in
                                fri)  r=$(calibrate_fri          "$lambda" "$b") ;;
                                stir) r=$(calibrate_stir_uniform "$lambda" "$b") ;;
                            esac
                            run_cell "$air" "$ldt" "$k" "$b" "$r" \
                                "$nist" "$lambda" "$q_log" "$hash" "$ext_e" \
                                "$mldsa" "$extra" "$cell_note"
                        done
                    done
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
