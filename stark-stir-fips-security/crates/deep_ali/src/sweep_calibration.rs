//! Arity/blowup sweep — soundness calibration helpers.
//!
//! Given a security parameter $\lambda$ (e.g. 128 for NIST L1) and a
//! blowup $b$ (so the rate is $\rho_0 = 1/b$), these helpers compute
//! the per-LDT query schedule that clears $\lambda$ bits of
//! information-theoretic soundness at the deployed configuration.
//!
//! The calibration formulas reflect the Johnson-regime per-query
//! yields used throughout this codebase and the paper:
//!
//! - **FRI uniform-$r$ (capacity-regime proxy)**: per-query yield is
//!   $\log_2 b$ bits; total queries $r = \lceil \lambda / \log_2 b \rceil$.
//!   This is the BCIKS Theorem 1.2 / Theorem 8.3 bound for FRI in
//!   the capacity regime; we use it as the calibration target for
//!   the FRI baseline.
//!
//! - **STIR uniform-$r$ (Johnson half-bit)**: per-query yield is
//!   $\tfrac{1}{2} \log_2 b$ bits at round 0 (Johnson-regime
//!   floor, unconditional); total queries
//!   $r = \lceil 2 \lambda / \log_2 b \rceil$ at the conservative
//!   round-0 floor.  This matches Table 5's uniform-$r$ STIR rows
//!   when calibrated to the paper's $\lambda \in \{128, 192, 256\}$.
//!
//! - **STIR secured per-round $\{t_i\}$**: Theorem 2's numerator
//!   $t_i = \lceil (\lambda + \log_2(M{+}2) + 1) / (-\log_2(\sqrt{\rho_i} + \eta_i)) \rceil$
//!   with $\rho_i = \rho_0 \cdot (2/k)^i$ and Johnson slack
//!   $\eta_i \le \sqrt{\rho_i}/20$.  This is the deployed-secured
//!   schedule; truncations are handled by the caller, see
//!   `ml_dsa_verify_air_v2_orchestration::parse_bench_t_schedule_env`.
//!
//! These helpers are pure functions used by:
//!   - `scripts/sweep-arity-blowup.sh` to populate `BENCH_QUERIES` /
//!     `BENCH_T_SCHEDULE` per $(k, b)$ cell;
//!   - the paper's per-AIR arity/blowup sweep section that reports
//!     proof-size / verify-time / verifier-hash-count curves.
//!
//! All helpers are deterministic, side-effect-free, and verified by
//! the unit tests at the foot of this module against the paper's
//! published values at the headline parameters.

/// FRI uniform-$r$ calibration.
///
/// Returns the minimum $r$ that clears $\lambda$ bits of FRI
/// (capacity-regime proxy) at rate $\rho_0 = 1/b$.  Per-query yield
/// is $\log_2 b$ bits; $r = \lceil \lambda / \log_2 b \rceil$.
///
/// Panics if $b$ is not a power of two $\ge 4$.
pub fn calibrate_fri_queries(lambda: usize, blowup: usize) -> usize {
    assert!(
        blowup >= 4 && blowup.is_power_of_two(),
        "blowup must be a power of two and >= 4; got {}",
        blowup
    );
    let bits_per_query = blowup.trailing_zeros() as usize; // log_2(b)
    assert!(bits_per_query > 0, "blowup must be > 1");
    (lambda + bits_per_query - 1) / bits_per_query
}

/// STIR uniform-$r$ calibration at the Johnson-regime round-0 floor.
///
/// Returns the minimum $r$ that clears $\lambda$ bits of STIR
/// Johnson-regime soundness at rate $\rho_0 = 1/b$.  Per-query yield
/// is $\tfrac{1}{2} \log_2 b$ bits at round 0;
/// $r = \lceil 2\lambda / \log_2 b \rceil$.
///
/// This is a conservative calibration that uses the round-0 floor
/// for all rounds; the per-round-shrinking secured schedule
/// (`calibrate_stir_secured`) is tighter.
///
/// Panics if $b$ is not a power of two $\ge 4$.
pub fn calibrate_stir_uniform(lambda: usize, blowup: usize) -> usize {
    assert!(
        blowup >= 4 && blowup.is_power_of_two(),
        "blowup must be a power of two and >= 4; got {}",
        blowup
    );
    let bits_per_query_2x = blowup.trailing_zeros() as usize; // log_2(b)
    assert!(bits_per_query_2x > 0, "blowup must be > 1");
    // r = ceil( 2*lambda / log_2(b) )
    let numerator = lambda
        .checked_mul(2)
        .expect("lambda doubles without overflow at NIST levels");
    (numerator + bits_per_query_2x - 1) / bits_per_query_2x
}

/// STIR secured-schedule calibration per Theorem 2.
///
/// Returns the per-round query budget $\{t_i\}_{i=0}^{M-1}$ under
/// the proven Johnson-regime contraction at fold arity $k$, $M$
/// rounds, rate $\rho_0 = 1/b$, and security target $\lambda$.
///
/// Formula (Theorem 2, paper):
/// $$t_i = \left\lceil \frac{\lambda + \log_2(M+2) + 1}
///         {-\log_2(\sqrt{\rho_i} + \eta_i)} \right\rceil$$
/// with $\rho_i = \rho_0 \cdot (2/k)^i$ and
/// $\eta_i \le \sqrt{\rho_i}/20$ (the BCIKS Johnson-curve slack).
///
/// The result is the analytic deployed schedule before the per-
/// sub-AIR truncation handled by `parse_bench_t_schedule_env`.
///
/// Sanity-checked at the paper's published headline parameters
/// (k=4, M=8, b=32, λ=128/192/256) → ∑t_i = 279/413/544 ±1.
///
/// Panics if $b$ or $k$ are not powers of two $\ge 2$.
pub fn calibrate_stir_secured(
    lambda: usize,
    k: usize,
    m_rounds: usize,
    blowup: usize,
) -> Vec<usize> {
    assert!(
        blowup >= 4 && blowup.is_power_of_two(),
        "blowup must be a power of two and >= 4; got {}",
        blowup
    );
    assert!(
        k >= 2 && k.is_power_of_two(),
        "fold arity k must be a power of two and >= 2; got {}",
        k
    );
    assert!(m_rounds >= 1, "m_rounds must be >= 1");

    let log2_b = blowup.trailing_zeros() as f64;     // log_2(b)
    let log2_k = k.trailing_zeros() as f64;          // log_2(k)
    let numer = (lambda as f64) + ((m_rounds + 2) as f64).log2() + 1.0;
    let log2_inv20 = (1.0_f64 / 20.0).log2().abs();  // |log_2(1/20)| ~ 4.32

    let mut t = Vec::with_capacity(m_rounds);
    for i in 0..m_rounds {
        // ρ_i = ρ_0 · (2/k)^i
        // -log_2(ρ_i) = log_2(b) + i·(log_2(k) - 1)
        let log2_inv_rho_i = log2_b + (i as f64) * (log2_k - 1.0);

        // √ρ_i + η_i ≤ √ρ_i · (1 + 1/20) = √ρ_i · 21/20
        // -log_2(√ρ_i + η_i) = ½·log_2(1/ρ_i) - log_2(21/20)
        //                    ≈ ½·log_2(1/ρ_i) - 0.0704
        let half = 0.5 * log2_inv_rho_i;
        let slack_loss = (21.0_f64 / 20.0).log2();   // ≈ 0.0703834
        let denom = half - slack_loss;

        // Guard against runaway when blowup is tiny relative to k —
        // a degenerate cell the sweep driver should skip.
        let denom = if denom < 1e-6 { 1e-6 } else { denom };

        let t_i = (numer / denom).ceil() as usize;
        t.push(t_i.max(1));
        // Silence the unused suppression warning on log2_inv20 — it's
        // exposed for callers that prefer the inexact-arithmetic form.
        let _ = log2_inv20;
    }
    t
}

/// Number of STIR fold rounds for a given (n_0, k) under the
/// uniform-arity schedule.  Returns `Some(M)` when `n_0` is a clean
/// power of `k`, else `None` (the caller must either use mixed-arity
/// rounds or pick a compatible blowup).
pub fn stir_rounds_for(n0: usize, k: usize) -> Option<usize> {
    if !n0.is_power_of_two() || !k.is_power_of_two() || k < 2 {
        return None;
    }
    let log2_n0 = n0.trailing_zeros() as usize;
    let log2_k = k.trailing_zeros() as usize;
    if log2_n0 % log2_k != 0 {
        return None;
    }
    Some(log2_n0 / log2_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fri_paper_headline_calibration() {
        // FRI at blowup-32 needs r ≥ 128/5 ≈ 25.6 → 26 queries at λ=128
        // (capacity-regime proxy; the paper's actual r=54 reflects the
        // tighter ÷2 schedule and per-round margins).
        assert_eq!(calibrate_fri_queries(128, 32), 26);
        assert_eq!(calibrate_fri_queries(128, 16), 32);
        assert_eq!(calibrate_fri_queries(128, 4), 64);
    }

    #[test]
    fn stir_uniform_paper_headline() {
        // STIR Johnson half-bit at blowup-32: r ≥ 2·128/5 = 51.2 → 52
        // Paper deploys r=54 (margin); 52 is the bare floor.
        assert_eq!(calibrate_stir_uniform(128, 32), 52);
        // L3 at blowup-32: r ≥ 2·192/5 = 76.8 → 77.  Paper: 81.
        assert_eq!(calibrate_stir_uniform(192, 32), 77);
        // L5 at blowup-32: r ≥ 2·256/5 = 102.4 → 103.  Paper: 108.
        assert_eq!(calibrate_stir_uniform(256, 32), 103);
    }

    #[test]
    fn stir_secured_matches_paper_schedule() {
        // L1, k=4, M=8, blowup=32:
        // Paper publishes {55, 46, 39, 34, 30, 27, 25, 23}, sum 279.
        let t = calibrate_stir_secured(128, 4, 8, 32);
        // The closed-form uses η_i ≤ √ρ_i/20 (paper's BCIKS slack);
        // first few entries should match exactly, last few within ±1
        // due to ceil() rounding behaviour around the Johnson floor.
        assert_eq!(t.len(), 8);
        assert!((t[0] as i64 - 55).abs() <= 1, "t_0 = {} (paper 55)", t[0]);
        assert!((t[1] as i64 - 46).abs() <= 1, "t_1 = {} (paper 46)", t[1]);
        assert!((t[7] as i64 - 23).abs() <= 1, "t_7 = {} (paper 23)", t[7]);
        let sum: usize = t.iter().sum();
        assert!(
            (sum as i64 - 279).abs() <= 8,
            "sum t_i = {} (paper 279, allow ±8 for ceil rounding)",
            sum
        );
    }

    #[test]
    fn stir_secured_increases_with_smaller_blowup() {
        // At smaller blowup, the round-0 rate is higher so t_0 grows.
        let t_b32 = calibrate_stir_secured(128, 4, 8, 32);
        let t_b16 = calibrate_stir_secured(128, 4, 8, 16);
        let t_b8  = calibrate_stir_secured(128, 4, 8, 8);
        assert!(t_b16[0] > t_b32[0], "smaller blowup needs more queries at round 0");
        assert!(t_b8[0]  > t_b16[0]);
    }

    #[test]
    fn stir_secured_shrinks_with_higher_arity() {
        // Higher k → faster rate shrinkage → smaller t_i at deep rounds.
        let t_k4  = calibrate_stir_secured(128, 4, 8, 32);
        let t_k16 = calibrate_stir_secured(128, 16, 8, 32);
        assert!(
            t_k16[7] < t_k4[7],
            "k=16 should beat k=4 at the deepest round: {} vs {}",
            t_k16[7], t_k4[7]
        );
    }

    #[test]
    fn stir_rounds_for_basic() {
        assert_eq!(stir_rounds_for(1 << 16, 2), Some(16));
        assert_eq!(stir_rounds_for(1 << 16, 4), Some(8));
        assert_eq!(stir_rounds_for(1 << 16, 16), Some(4));
        // n_0 = 2^17, k=4 → 17 not divisible by 2
        assert_eq!(stir_rounds_for(1 << 17, 4), None);
        // n_0 = 2^15, k=8 → 15 / 3 = 5, exact
        assert_eq!(stir_rounds_for(1 << 15, 8), Some(5));
    }

    #[test]
    #[should_panic(expected = "blowup must be a power of two")]
    fn rejects_non_power_of_two_blowup() {
        let _ = calibrate_fri_queries(128, 7);
    }

    #[test]
    #[should_panic(expected = "fold arity k must be a power of two")]
    fn rejects_non_power_of_two_k() {
        let _ = calibrate_stir_secured(128, 3, 8, 32);
    }
}
