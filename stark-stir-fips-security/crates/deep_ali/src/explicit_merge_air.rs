//! P5.1 — AIR-evaluator scaffolding for the explicit merge.
//!
//! Integration plumbing that lets the explicit-merge prover (and the
//! verifier) talk to an AIR through a single trait, without the
//! merge layer learning about the constraint algebra of any
//! particular AIR (ML-DSA-v2, RSA-2048, ECDSA-p256, …).
//!
//! ## The trait
//!
//! `AirOodEvaluator<E>` exposes two responsibilities of an AIR at
//! the OOD layer:
//!
//!   1. The shift set `Σ = {σ_0, …, σ_{k-1}}` referenced by the
//!      constraint composition (`Σ = {1, ω}` for first-order
//!      transition systems, longer for higher-order recurrences).
//!
//!   2. Evaluation of the FS-aggregated constraint polynomial at
//!      the OOD point: given the claimed trace evaluations
//!      `T̂_col(σ_i z)` and the OOD point `z`, return
//!      `Σ_j α_j Φ_j(z) ∈ E`.
//!
//! The `α_j` FS coefficients are AIR-internal and assumed to be
//! either (a) constants the AIR knows, or (b) derived from the
//! transcript at AIR-construction time and embedded in the
//! implementation.
//!
//! ## Helpers
//!
//! - `eval_h0_polynomial_at_ext`: evaluate a polynomial given by its
//!   evaluations on a multiplicative subgroup `H_0` (size `n`, a
//!   power of two) at an arbitrary extension point `z ∈ E`.  Uses
//!   `ark_poly`'s IFFT to extract F-coefficients, then Horner's
//!   method in `E`.
//!
//! - `build_ood_claims_from_witness`: given `(witness, air, z)`,
//!   produce the `OodClaims<E>` the prover sends on the wire — i.e.
//!   `T̂_col(σ_i z)` for each `(col, i)` and `q̂(z)`.
//!
//! ## Scope
//!
//! Primitives + a "constant-boundary" toy AIR fixture for tests.
//! Wiring into `fri.rs` / `stir_halve.rs` is deferred to P5.2.

use ark_ff::Field;
use ark_goldilocks::Goldilocks as F;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

use crate::explicit_merge::{MergeWitness, OodClaims};
use crate::tower_field::TowerField;

// ─── Trait ───────────────────────────────────────────────────────

/// The minimal AIR surface seen by the merge layer.
///
/// Implementors describe:
///   - the shift set `Σ` over `F` (length `k`),
///   - how to evaluate the FS-aggregated constraint
///     `Σ_j α_j Φ_j(z)` at an OOD point, given the claimed trace
///     evaluations `T̂_col(σ_i z)`.
///
/// The trait is intentionally narrow — `Φ_j` are not exposed
/// individually; only the aggregated `Φ(z)` matters at the OOD
/// layer.  This keeps the merge layer AIR-agnostic.
pub trait AirOodEvaluator<E: TowerField> {
    /// Number of trace columns (`w`).
    fn n_columns(&self) -> usize;

    /// Shift set `Σ = {σ_0, …, σ_{k-1}}` over the base field.
    fn shifts(&self) -> &[F];

    /// Evaluate the FS-aggregated constraint polynomial
    /// `Σ_j α_j Φ_j(z)` at the OOD point `z`, given the claimed
    /// `T̂_col(σ_i z)` table.
    ///
    /// `trace_at_shifts[col][shift_idx]` is shaped identically to
    /// `OodClaims::trace_at_shifts`.
    fn constraint_at_z(&self, z: E, trace_at_shifts: &[Vec<E>]) -> E;
}

// ─── Polynomial evaluation at an extension-field point ───────────

/// Evaluate a polynomial given by its evaluations on the
/// multiplicative subgroup `H_0` of order `n = evals_on_h0.len()`
/// (a power of two) at an arbitrary extension point `z ∈ E`.
///
/// Algorithm: IFFT to extract F-coefficients `c_0, …, c_{n-1}`,
/// then Horner's method in `E` to evaluate `Σ c_i z^i`.
///
/// Cost: `O(n log n)` field ops for the IFFT plus `O(n)` extension
/// multiplications for the Horner pass.
///
/// Panics if `n` is not a power of two (the ark_poly domain
/// constructor returns `None`).
pub fn eval_h0_polynomial_at_ext<E: TowerField>(
    evals_on_h0: &[F],
    point: E,
) -> E {
    let n = evals_on_h0.len();
    let domain = GeneralEvaluationDomain::<F>::new(n)
        .expect("eval_h0_polynomial_at_ext: |H_0| must be a power of two");
    let coeffs: Vec<F> = domain.ifft(evals_on_h0);
    horner_ext::<E>(&coeffs, point)
}

/// Horner's-method polynomial evaluation: `Σ c_i · z^i` for
/// F-coefficients evaluated at `z ∈ E`.
fn horner_ext<E: TowerField>(coeffs: &[F], z: E) -> E {
    let mut acc = E::zero();
    for &c in coeffs.iter().rev() {
        acc = acc * z + E::from_fp(c);
    }
    acc
}

// ─── Build OodClaims from witness + AIR ──────────────────────────

/// Build `OodClaims<E>` from `(witness, air, z)`.
///
/// For each column `col ∈ [0, w)` and shift `σ_i ∈ Σ`, evaluates the
/// trace polynomial encoded in `witness.trace_columns[col]` at the
/// shifted OOD point `σ_i · z ∈ E`.  Also evaluates the ALI quotient
/// `witness.ali_quotient` at `z` to get `q̂(z)`.
///
/// Assumes `witness.trace_columns[col]` is the evaluation of the
/// trace polynomial on a multiplicative-subgroup `H_0` of order
/// `witness.trace_columns[col].len()` (power of two).
///
/// Panics if `air.n_columns() != witness.trace_columns.len()`.
pub fn build_ood_claims_from_witness<E, A>(
    witness: &MergeWitness,
    air: &A,
    z: E,
) -> OodClaims<E>
where
    E: TowerField,
    A: AirOodEvaluator<E>,
{
    let w = witness.trace_columns.len();
    assert_eq!(
        air.n_columns(), w,
        "AIR n_columns ({}) does not match witness trace_columns ({})",
        air.n_columns(), w,
    );

    let shifts = air.shifts();
    let k = shifts.len();
    let trace_at_shifts: Vec<Vec<E>> = (0..w)
        .map(|col| {
            (0..k)
                .map(|i| {
                    let shift_z = E::from_fp(shifts[i]) * z;
                    eval_h0_polynomial_at_ext::<E>(&witness.trace_columns[col], shift_z)
                })
                .collect()
        })
        .collect();

    let q_at_z = eval_h0_polynomial_at_ext::<E>(&witness.ali_quotient, z);
    OodClaims { z, trace_at_shifts, q_at_z }
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit_merge_ood_check::{check_ood_consistency_from_claims, vanishing_at_ext};
    use crate::sextic_ext::SexticExt;
    use ark_ff::{FftField, One, UniformRand, Zero};
    use rand::{rngs::StdRng, SeedableRng};

    type Ext = SexticExt;

    fn ext_random(rng: &mut StdRng) -> Ext {
        Ext::from_fp_components(&[
            F::rand(rng), F::rand(rng), F::rand(rng),
            F::rand(rng), F::rand(rng), F::rand(rng),
        ]).expect("ext components")
    }

    // ── Horner sanity ──

    #[test]
    fn horner_ext_matches_naive_eval_at_random_point() {
        let mut rng = StdRng::seed_from_u64(0x5101_0001);
        let n = 8usize;
        let coeffs: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let z = ext_random(&mut rng);
        let horner = horner_ext::<Ext>(&coeffs, z);

        let mut naive = Ext::zero();
        let mut zk = Ext::one();
        for &c in &coeffs {
            naive += Ext::from_fp(c) * zk;
            zk = zk * z;
        }
        assert_eq!(horner, naive);
    }

    // ── eval_h0_polynomial_at_ext sanity ──

    /// The helper should return the IFFT-extracted polynomial
    /// evaluated at `z`.  We compute that two ways: via the helper,
    /// and via a manual `Σ c_i z^i` on the IFFT coefficients.
    #[test]
    fn eval_h0_polynomial_at_ext_matches_manual_horner() {
        let mut rng = StdRng::seed_from_u64(0x5102_0002);
        let n = 16usize;
        let evals: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let z = ext_random(&mut rng);

        let via_helper = eval_h0_polynomial_at_ext::<Ext>(&evals, z);
        let coeffs = GeneralEvaluationDomain::<F>::new(n).unwrap().ifft(&evals);
        let manual = horner_ext::<Ext>(&coeffs, z);
        assert_eq!(via_helper, manual);
    }

    /// At any `x ∈ H_0`, the helper should reproduce `evals_on_h0[i]`
    /// (when called at the lifted base-field value).
    #[test]
    fn eval_h0_polynomial_at_ext_recovers_evals_at_subgroup_points() {
        let mut rng = StdRng::seed_from_u64(0x5103_0003);
        let n = 16usize;
        let omega: F = F::get_root_of_unity(n as u64).expect("two-adic root");
        let h0: Vec<F> = (0..n).map(|i| omega.pow_u64(i as u64)).collect();
        let evals: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        for i in 0..n {
            let at = Ext::from_fp(h0[i]);
            let got = eval_h0_polynomial_at_ext::<Ext>(&evals, at);
            let want = Ext::from_fp(evals[i]);
            assert_eq!(got, want, "subgroup recovery failed at i={}", i);
        }
    }

    // ── Toy "constant boundary" AIR fixture ──
    //
    // AIR: single column.  Shift set Σ = {1} (no transitions).
    // Constraint: Φ(X) = T(X) - c.  Vanishes on H iff T is the
    // constant c on H.
    //
    // Honest witness construction:
    //   1. Pick a random Q on H_0 (degree ≤ d_0 ≤ T-1 — we keep Q
    //      bounded by T-1 by drawing from the IFFT of length-T evals,
    //      then re-evaluating on H_0).
    //   2. Compute Z_H(X) = X^T - 1 on H_0.
    //   3. Set T(X) := Q(X) · Z_H(X) + c on H_0.
    //   On H ⊂ H_0, Z_H = 0 so T = c.  Off H, T is non-trivial.

    struct ConstantBoundaryAir { c: F }

    impl AirOodEvaluator<Ext> for ConstantBoundaryAir {
        fn n_columns(&self) -> usize { 1 }
        fn shifts(&self) -> &[F] {
            // 'static slice — we use the multiplicative identity.
            const ONES: [F; 1] = [F::ONE];
            &ONES
        }
        fn constraint_at_z(&self, _z: Ext, trace_at_shifts: &[Vec<Ext>]) -> Ext {
            // Φ(z) = T̂(z) - c.
            trace_at_shifts[0][0] - Ext::from_fp(self.c)
        }
    }

    /// Build an honest witness for the constant-boundary AIR.
    fn build_honest_constant_boundary_witness(
        rng: &mut StdRng,
        trace_len: usize,
        blowup: usize,
        c: F,
    ) -> (MergeWitness, Vec<F>) {
        let n = trace_len * blowup;
        let omega_n: F = F::get_root_of_unity(n as u64).expect("two-adic root");
        let h0: Vec<F> = (0..n).map(|i| omega_n.pow_u64(i as u64)).collect();

        // 1. Q on H_0: pick T length-T random evals, IFFT, re-evaluate on H_0.
        let q_t_evals: Vec<F> = (0..trace_len).map(|_| F::rand(rng)).collect();
        let q_domain = GeneralEvaluationDomain::<F>::new(trace_len).unwrap();
        let q_coeffs: Vec<F> = q_domain.ifft(&q_t_evals);
        let q_on_h0: Vec<F> = h0.iter()
            .map(|&x| {
                let mut acc = F::zero();
                for &cc in q_coeffs.iter().rev() {
                    acc = acc * x + cc;
                }
                acc
            }).collect();

        // 2. Z_H(x) = x^T - 1 on H_0.
        let zh_on_h0: Vec<F> = h0.iter()
            .map(|&x| x.pow([trace_len as u64]) - F::one())
            .collect();

        // 3. T(x) = Q(x) · Z_H(x) + c on H_0.
        let t_on_h0: Vec<F> = q_on_h0.iter().zip(zh_on_h0.iter())
            .map(|(&q, &zh)| q * zh + c)
            .collect();

        // Blinder: random.
        let r: Vec<F> = (0..n).map(|_| F::rand(rng)).collect();

        // d_c = 2 (constraint degree is 1 in T, but we keep margin for d_0).
        let d_c = 2usize;
        let d0 = (d_c - 1) * trace_len - 1;
        let k_shifts = 1usize;

        let w = MergeWitness {
            trace_columns: vec![t_on_h0],
            ali_quotient: q_on_h0,
            blinder: r,
            trace_len, d0, k_shifts,
        };
        (w, h0)
    }

    /// End-to-end round-trip: build honest witness, build OOD claims
    /// via the AIR-evaluator path, check OOD consistency holds.
    #[test]
    fn constant_boundary_air_roundtrips_through_check_ood_consistency() {
        let mut rng = StdRng::seed_from_u64(0x5110_0010);
        let trace_len = 8usize;
        let blowup = 4usize;
        let c = F::from(7u64);

        let (witness, _h0) = build_honest_constant_boundary_witness(
            &mut rng, trace_len, blowup, c);

        let air = ConstantBoundaryAir { c };

        // Pick OOD point.
        let z = ext_random(&mut rng);

        // Build OOD claims via the trait + helper.
        let ood = build_ood_claims_from_witness::<Ext, _>(&witness, &air, z);

        // AIR-aware Φ̂(z) computation.
        let phi_at_z = air.constraint_at_z(z, &ood.trace_at_shifts);

        // Sanity: Φ̂(z) = T̂(z) - c and T̂(z) = Q̂(z) · Z_H(z) + c, so
        // Φ̂(z) = Q̂(z) · Z_H(z).
        let z_h = vanishing_at_ext::<Ext>(z, trace_len);
        assert_eq!(phi_at_z, ood.q_at_z * z_h,
            "boundary-AIR algebraic identity broke (Φ = Q · Z_H)");

        // OOD consistency check.
        assert!(check_ood_consistency_from_claims::<Ext>(&ood, phi_at_z, trace_len),
            "honest boundary-AIR witness must pass the OOD consistency check");
    }

    /// Tamper: corrupting a single H_0-evaluation of T (after honest
    /// build) shifts T̂(z) and (since Q is unchanged) breaks the
    /// `Φ = Q · Z_H` identity.  OOD check must reject.
    #[test]
    fn tampering_t_after_honest_build_breaks_ood_check() {
        let mut rng = StdRng::seed_from_u64(0x5111_0011);
        let trace_len = 8usize;
        let blowup = 4usize;
        let c = F::from(11u64);

        let (mut witness, _h0) = build_honest_constant_boundary_witness(
            &mut rng, trace_len, blowup, c);
        let air = ConstantBoundaryAir { c };

        witness.trace_columns[0][3] += F::one();

        let z = ext_random(&mut rng);
        let ood = build_ood_claims_from_witness::<Ext, _>(&witness, &air, z);
        let phi_at_z = air.constraint_at_z(z, &ood.trace_at_shifts);

        assert!(!check_ood_consistency_from_claims::<Ext>(&ood, phi_at_z, trace_len),
            "tampered-T witness must fail the OOD consistency check");
    }

    /// AIR / witness column-count mismatch panics.
    #[test]
    #[should_panic(expected = "AIR n_columns")]
    fn ncolumns_mismatch_panics() {
        let mut rng = StdRng::seed_from_u64(0x5112_0012);
        let trace_len = 8usize;
        let blowup = 4usize;
        let (mut witness, _h0) = build_honest_constant_boundary_witness(
            &mut rng, trace_len, blowup, F::from(3u64));

        // Add a spurious extra column.
        witness.trace_columns.push(vec![F::one(); trace_len * blowup]);

        let air = ConstantBoundaryAir { c: F::from(3u64) };
        let z = ext_random(&mut rng);
        let _ = build_ood_claims_from_witness::<Ext, _>(&witness, &air, z);
    }
}
