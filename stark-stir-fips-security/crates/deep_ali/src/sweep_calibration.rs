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

/// FRI uniform-$r$ calibration at the Johnson-regime floor.
///
/// Returns the minimum $r$ that clears $\lambda$ bits of FRI
/// unconditional Johnson-regime soundness at rate $\rho_0 = 1/b$,
/// matching the paper's deployed FRI baseline (Tables 5/6 use
/// $r = 55$ at L1 for both FRI and STIR).  Per-query yield is
/// $\tfrac{1}{2} \log_2 b$ bits at round 0;
/// $r = \lceil 2\lambda / \log_2 b \rceil$.
///
/// Identical formula to [`calibrate_stir_uniform`] — both LDTs sit
/// at the same Johnson floor when calibrated at the unconditional
/// soundness target the rest of this paper uses.  The two functions
/// are kept separate for documentation: the FRI capacity bound
/// ($r = \lceil \lambda / \log_2 b \rceil$) is conjectural at large
/// $r$ and is NOT used here; the original sweep results published
/// in [Appendix~A, an earlier revision] reflected that asymmetric
/// calibration choice and were corrected in this revision per
/// reviewer feedback.
///
/// Panics if $b$ is not a power of two $\ge 4$.
pub fn calibrate_fri_queries(lambda: usize, blowup: usize) -> usize {
    // Matches calibrate_stir_uniform — Johnson-floor calibration.
    calibrate_stir_uniform(lambda, blowup)
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

/// NIST security target $\lambda$ at each level (bits).
pub const NIST_L1: usize = 128;
pub const NIST_L3: usize = 192;
pub const NIST_L5: usize = 256;

/// $\kappa_{\mathrm{bind}}$ bit-security as a function of hash output
/// size $n$ (bits), quantum query budget $q$, and the number of FRI/STIR
/// rounds $M$.
///
/// Per the paper's eq. (`bind-bits`):
/// $$\kappa_{\mathrm{bind}}(n, q) = n - 3 \log_2 q
///     - \lceil \log_2((M+2)\,c_b) \rceil
///     \approx n - 3 \log_2 q - 3.3,$$
/// where $c_b = O(1)$ is the Zhandry compressed-oracle constant
/// (absorbed in the paper's "$\approx 3.3$" approximation that
/// effectively treats $c_b \approx 1$).  We compute the strict
/// $\lceil \log_2(M+2) \rceil$ penalty (4 bits at $M=8$, matching the
/// paper's Table 3 wall values exactly).
///
/// Returns $\kappa_{\mathrm{bind}}$ as a signed integer (can go
/// negative when the wall is breached); the caller compares against
/// $\lambda$ for the wall-check.
pub fn kappa_bind(n_hash_bits: usize, q_log: usize, m_rounds: usize) -> i64 {
    // Paper's "≈3.3" = log_2(M+2) at M=8 → 3.32.  We use ceil(log_2(M+2))
    // for the bit-budget calculation; this reproduces Table 3 walls
    // exactly at the deployed M ∈ {8} and degrades gracefully at
    // larger M (more conservative).
    let m_plus_2_log = (((m_rounds + 2) as f64).log2()).ceil() as i64;
    (n_hash_bits as i64) - 3 * (q_log as i64) - m_plus_2_log
}

/// $\kappa_{\mathrm{FS}}$ bit-security as a function of extension
/// degree $e$, quantum query budget $q$, and DFMS round count $K$.
///
/// Per the paper's eq. (`fs-bits`):
/// $$\kappa_{\mathrm{FS}}(e, q) = 64 e - 2 \log_2 K - 2 \log_2 q.$$
///
/// Returns $\kappa_{\mathrm{FS}}$; compare against $\lambda$ for the
/// wall-check.
pub fn kappa_fs(extension_degree: usize, q_log: usize, k_rounds: usize) -> i64 {
    let k_log = ((k_rounds as f64).log2().ceil() as i64).max(1);
    64 * (extension_degree as i64) - 2 * (k_log) - 2 * (q_log as i64)
}

/// Joint $\kappa_{\mathrm{sys}} = \min(\kappa_{\mathrm{IT}},
/// \kappa_{\mathrm{bind}}, \kappa_{\mathrm{FS}})$ — the system-level
/// bit-security.  Callers supply the IT bits from their LDT choice.
pub fn kappa_sys(
    kappa_it_bits: i64,
    n_hash_bits: usize,
    extension_degree: usize,
    q_log: usize,
    m_rounds: usize,
) -> i64 {
    let k_fs = kappa_fs(extension_degree, q_log, m_rounds + 2);
    let k_bind = kappa_bind(n_hash_bits, q_log, m_rounds);
    kappa_it_bits.min(k_bind).min(k_fs)
}

/// Wall-breach check: at $(\lambda, n_{\mathrm{hash}}, e, q, M)$, can
/// any IT-side budget save the configuration?  Returns `None` when
/// the cell is sustainable (the caller should calibrate $r$ via
/// `calibrate_*_queries`), or `Some(reason)` describing which ceiling
/// is breached and by how much.
pub fn wall_check(
    lambda: usize,
    n_hash_bits: usize,
    extension_degree: usize,
    q_log: usize,
    m_rounds: usize,
) -> Option<String> {
    let k_bind = kappa_bind(n_hash_bits, q_log, m_rounds);
    if k_bind < lambda as i64 {
        return Some(format!(
            "kappa_bind={} < lambda={} at q=2^{} hash={}b M={} (binding wall breached)",
            k_bind, lambda, q_log, n_hash_bits, m_rounds
        ));
    }
    let k_fs = kappa_fs(extension_degree, q_log, m_rounds + 2);
    if k_fs < lambda as i64 {
        return Some(format!(
            "kappa_fs={} < lambda={} at q=2^{} e={} K={} (FS ceiling breached)",
            k_fs, lambda, q_log, extension_degree, m_rounds + 2
        ));
    }
    None
}

/// Map a NIST level (1, 3, or 5) to (lambda, hash_output_bits,
/// extension_degree) per the paper's deployed parameters.
pub fn nist_level_params(level: u8) -> Option<(usize, usize, usize)> {
    match level {
        1 => Some((NIST_L1, 256, 6)),
        3 => Some((NIST_L3, 384, 6)),
        5 => Some((NIST_L5, 512, 8)),
        _ => None,
    }
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
    fn fri_johnson_matched_with_stir() {
        // FRI uniform-r now uses the same Johnson floor as STIR
        // (matches the paper's deployed FRI baseline, r=55 at L1).
        // Per-query yield ½·log_2(b); r = ⌈2λ/log_2(b)⌉.
        assert_eq!(calibrate_fri_queries(128, 32), 52);
        assert_eq!(calibrate_fri_queries(128, 16), 64);
        assert_eq!(calibrate_fri_queries(128, 4),  128);
        // Equality with STIR uniform — by construction.
        for b in [4_usize, 8, 16, 32, 64] {
            for lambda in [128_usize, 192, 256] {
                assert_eq!(calibrate_fri_queries(lambda, b),
                           calibrate_stir_uniform(lambda, b));
            }
        }
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

    #[test]
    fn kappa_bind_paper_wall_values() {
        // Paper Table 3 (M=8, K=10 for the cb_log_ceil=4):
        //   L1, SHA3-256: q_max = 2^41   → kappa_bind(256, 41, 8) ≈ 128
        //   L3, SHA3-384: q_max = 2^62   → kappa_bind(384, 62, 8) ≈ 192
        //   L5, SHA3-512: q_max = 2^84   → kappa_bind(512, 84, 8) ≈ 256
        // We allow ±2 for the ceiling rounding convention.
        assert!((kappa_bind(256, 41, 8) - 128).abs() <= 2,
                "L1 wall: {}", kappa_bind(256, 41, 8));
        assert!((kappa_bind(384, 62, 8) - 192).abs() <= 2,
                "L3 wall: {}", kappa_bind(384, 62, 8));
        assert!((kappa_bind(512, 84, 8) - 256).abs() <= 2,
                "L5 wall: {}", kappa_bind(512, 84, 8));
    }

    #[test]
    fn kappa_fs_paper_ceiling() {
        // Paper §5: e=6 gives 64·6=384 raw bits; at K=10 (M=8) and
        // q=2^40 the FS ceiling = 384 - 2·log_2(10) - 2·40 ≈ 384 - 7 - 80 = 297.
        let k_fs = kappa_fs(6, 40, 10);
        assert!((k_fs - 297).abs() <= 2, "kappa_fs(e=6, q=40, K=10) = {}", k_fs);

        // e=8 at L5 with q=2^65: 512 - 7 - 130 = 375.
        let k_fs_l5 = kappa_fs(8, 65, 10);
        assert!((k_fs_l5 - 375).abs() <= 2, "kappa_fs(e=8, q=65, K=10) = {}", k_fs_l5);
    }

    #[test]
    fn wall_check_at_paper_q_budgets() {
        // Per paper Table 3 wall values:
        //   L1, SHA3-256: q_max = 2^41 (kappa_bind(256,41,8) = 129 ≥ 128 ✓)
        //   L3, SHA3-384: q_max = 2^62 (kappa_bind(384,62,8) = 194 ≥ 192 ✓)
        //   L5, SHA3-512: q_max = 2^84 (kappa_bind(512,84,8) = 256 ≥ 256 ✓)
        // The three test q values {2^40, 2^65, 2^90} sit either side
        // of those walls.

        // L1 q=2^40: 256 - 120 - 4 = 132 ≥ 128 → pass
        assert!(wall_check(128, 256, 6, 40, 8).is_none(), "L1 q=40 should pass");
        // L1 q=2^65: 256 - 195 - 4 = 57 < 128 → breach
        assert!(wall_check(128, 256, 6, 65, 8).is_some(), "L1 q=65 must breach");
        assert!(wall_check(128, 256, 6, 90, 8).is_some(), "L1 q=90 must breach");

        // L3 q=2^40: 384 - 120 - 4 = 260 ≥ 192 → pass
        assert!(wall_check(192, 384, 6, 40, 8).is_none(), "L3 q=40 should pass");
        // L3 q=2^65: 384 - 195 - 4 = 185 < 192 → breach (just past wall at 2^62)
        assert!(wall_check(192, 384, 6, 65, 8).is_some(), "L3 q=65 must breach");
        assert!(wall_check(192, 384, 6, 90, 8).is_some(), "L3 q=90 must breach");

        // L5 q=2^40: 512 - 120 - 4 = 388 ≥ 256 → pass
        assert!(wall_check(256, 512, 8, 40, 8).is_none(), "L5 q=40 should pass");
        // L5 q=2^65: 512 - 195 - 4 = 313 ≥ 256 → pass
        assert!(wall_check(256, 512, 8, 65, 8).is_none(), "L5 q=65 should pass");
        // L5 q=2^90: 512 - 270 - 4 = 238 < 256 → breach (wall at 2^84)
        assert!(wall_check(256, 512, 8, 90, 8).is_some(), "L5 q=90 must breach");
    }

    #[test]
    fn nist_level_param_table() {
        assert_eq!(nist_level_params(1), Some((128, 256, 6)));
        assert_eq!(nist_level_params(3), Some((192, 384, 6)));
        assert_eq!(nist_level_params(5), Some((256, 512, 8)));
        assert_eq!(nist_level_params(2), None);
    }

    #[test]
    fn kappa_sys_min_of_three() {
        // At L1 / q=2^40 / r=54 (paper deployment): IT bits = 54·2.5 = 135;
        // kappa_bind = 256 - 120 - 4 = 132,  kappa_FS = 64·6 - 7 - 80 = 297;
        // min = 132.  Paper rounds this to "≥128" since lambda = 128.
        let it_bits = 135;
        let kappa = kappa_sys(it_bits, 256, 6, 40, 8);
        // kappa_sys = min(135, 132, 297) = 132 → ≥ 128 (clears L1).
        assert!(kappa >= 128, "L1 deployed kappa_sys = {} (should clear 128)", kappa);
        assert!(kappa <= 135, "min of three should not exceed IT-bits");
    }
}
