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

# ── NIST-level parameter table ───────────────────────────────────────
# Maps level → (lambda, hash_bits, ext_e, hash_feature, mldsa_feature,
# extra_features).  Matches `sweep_calibration::nist_level_params`.
params_for_level () {
    case "$1" in
        1) echo "128 256 6 sha3-256 mldsa-44 ''" ;;
        3) echo "192 384 6 sha3-384 mldsa-65 ''" ;;
        5) echo "256 512 8 sha3-512 mldsa-87 tower-octic" ;;
        *) echo "" ;;
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
    local mldsa=${11} extra=${12}
    local features prove_ms verify_ms proof_kib n_trace note=""
    local log="$RESULTS_DIR/sweep-${air}-${ldt}-k${k}-b${b}-L${nist}-q${q_log}.log"

    features=$(features_for "$air" "$hash" "$mldsa" "$extra")

    case "$air" in
        rsa2048)
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
    echo "${SWEEP_PHASE},${nist},${lambda},${q_log},${hash},${ext_e},${air},${ldt},${k},${b},${r},${prove_ms},${verify_ms},${proof_kib},${n_trace},${RAYON_NUM_THREADS},${note}" >> "$CSV"
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

# Parse comma-separated subsets.
IFS=',' read -ra NIST_LEVELS <<< "$SWEEP_NIST"
IFS=',' read -ra Q_LOGS <<< "$SWEEP_Q_LOGS"

echo "## Sweep started: $(date -Iseconds)"
echo "## Phase(s): ${PHASES[*]}"
echo "## NIST levels: ${NIST_LEVELS[*]}"
echo "## q_log: ${Q_LOGS[*]}"
echo "## dry-run=$SWEEP_DRY_RUN"
echo

for phase in "${PHASES[@]}"; do
    SWEEP_PHASE="$phase"
    echo "═════ Phase $phase ═════"
    for nist in "${NIST_LEVELS[@]}"; do
        # Look up level parameters.
        read -r lambda n_hash ext_e hash mldsa extra <<< "$(params_for_level "$nist")"
        if [ -z "$lambda" ]; then
            echo "skip: unknown NIST level $nist" >&2; continue
        fi
        echo "──── L${nist} (λ=${lambda}, ${hash}, F_p^${ext_e}) ────"
        for q_log in "${Q_LOGS[@]}"; do
            # M_rounds for the wall check.  We use the deployed M=8 here
            # since the schedule lengths in v2_fri_params depend on the
            # AIR's n_0; M=8 is the paper's deployed default.
            wall=$(wall_check "$lambda" "$n_hash" "$ext_e" "$q_log" 8)
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
                            if [ "$wall" != "ok" ]; then
                                record_wall_breach "$air" "$ldt" "$k" "$b" "$r" \
                                    "$nist" "$lambda" "$q_log" "$hash" "$ext_e" "$wall"
                                continue
                            fi
                            run_cell "$air" "$ldt" "$k" "$b" "$r" \
                                "$nist" "$lambda" "$q_log" "$hash" "$ext_e" \
                                "$mldsa" "$extra"
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
