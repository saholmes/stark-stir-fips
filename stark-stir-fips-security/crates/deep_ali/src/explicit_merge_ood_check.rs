//! P3 — Verifier-side OOD consistency check (eq:ali-check).
//!
//! Implements the algebraic gate that ties the OOD trace claims
//! `T̂(σ_i z)` (carried in `OodClaims::trace_at_shifts`) to the
//! ALI-quotient claim `Q̂(z)` (carried in `OodClaims::q_at_z`):
//!
//! ```text
//!     Σ_j  α_j · Φ_j(T̂(σ_1 z), …, T̂(σ_k z))   =?   Q̂(z) · Z_H(z)
//! ```
//!
//! where `H` is the trace-aligned domain of size `T = trace_len`,
//! assumed to be the multiplicative subgroup of that order, so
//! `Z_H(X) = X^T - 1`.
//!
//! ## AIR-agnostic at this layer
//!
//! The constraint polynomial `Φ_j` is AIR-specific (ML-DSA-v2's four
//! sub-AIRs differ from RSA-2048, ECDSA-p256, …).  We expose the check
//! as a primitive that takes the pre-aggregated constraint value
//! `Σ_j α_j Φ_j(z)` (the AIR-aware caller computes this from the OOD
//! trace claims and the FS-derived α coefficients) plus the
//! trace-domain size `T`, and checks the algebraic equality against
//! `Q̂(z)`.
//!
//! Keeping the AIR boundary outside this primitive matches the paper:
//! the merge construction is universal, only the Φ-evaluator differs
//! between AIRs.
//!
//! ## What this check buys you
//!
//! A passing `check_ood_consistency` is the FS-binding step at the
//! merge layer: it forces the prover to have committed to a `Q` such
//! that `Φ(X) = Q(X) · Z_H(X)` (i.e., the AIR constraints vanish on
//! `H`), up to the FS soundness error `ε_FS` of the underlying
//! STIR/FRI protocol, under the standard Schwartz–Zippel argument
//! over the large extension field `F_{p^e}` (see paper Theorem 1
//! Event E_1).
//!
//! ## Scope of this commit
//!
//! Primitive + tests.  Wiring into the STIR/FRI verifier proper is
//! deferred to a subsequent commit.

use crate::explicit_merge::OodClaims;
use crate::tower_field::TowerField;

// ─── Z_H(z) over the extension field ─────────────────────────────

/// Evaluate the vanishing polynomial of the trace-aligned domain `H`
/// at an extension-field point `z`.
///
/// Assumes `H` is the multiplicative subgroup of order `trace_len`,
/// so `Z_H(X) = X^{trace_len} - 1`.
///
/// At any `z ∉ H` this is non-zero (so dividing by it is safe), which
/// is the standard ALI sample-outside-the-domain condition the
/// transcript enforces on `z`.
///
/// Panics if `trace_len == 0` (no domain, ill-defined).
#[inline]
pub fn vanishing_at_ext<E: TowerField>(z: E, trace_len: usize) -> E {
    assert!(trace_len > 0, "vanishing_at_ext: trace_len must be > 0");
    z.pow_u64(trace_len as u64) - E::one()
}

// ─── eq:ali-check ────────────────────────────────────────────────

/// Verifier-side OOD consistency check.
///
/// Checks
/// ```text
///     constraint_at_z  =?  q_at_z · Z_H(z)
/// ```
/// where `Z_H(z) = z^trace_len - 1`.
///
/// `constraint_at_z` is `Σ_j α_j Φ_j(z)`, evaluated by the AIR-aware
/// caller from the OOD trace claims `T̂(σ_i z)` using the FS-derived
/// α coefficients.
///
/// This is a *constant-time* algebraic comparison; it does not by
/// itself do any branch or short-circuit, but a tampered claim
/// produces a non-equal pair w.h.p. over the FS draw of `z`.
#[inline]
pub fn check_ood_consistency<E: TowerField>(
    z: E,
    q_at_z: E,
    constraint_at_z: E,
    trace_len: usize,
) -> bool {
    let z_h = vanishing_at_ext::<E>(z, trace_len);
    constraint_at_z == q_at_z * z_h
}

/// Convenience wrapper that pulls `z` and `q_at_z` straight from
/// `OodClaims`.  Production verifiers use this; the bare
/// `check_ood_consistency` is exposed for cases where the caller
/// only has the raw scalars.
#[inline]
pub fn check_ood_consistency_from_claims<E: TowerField>(
    ood: &OodClaims<E>,
    constraint_at_z: E,
    trace_len: usize,
) -> bool {
    check_ood_consistency::<E>(ood.z, ood.q_at_z, constraint_at_z, trace_len)
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sextic_ext::SexticExt;
    use ark_ff::{FftField, One, UniformRand};
    use ark_goldilocks::Goldilocks as F;
    use rand::{rngs::StdRng, SeedableRng};

    type Ext = SexticExt;

    /// `Z_H(z) = z^T - 1` should equal the explicit product form
    /// `∏_{ω ∈ H} (z - ω)` for small `T`.
    #[test]
    fn vanishing_at_ext_matches_product_form() {
        let mut rng = StdRng::seed_from_u64(0x1234_5678);
        for &t in &[2usize, 4, 8, 16] {
            let omega: F = F::get_root_of_unity(t as u64).expect("two-adic root");
            let h: Vec<F> = (0..t).map(|i| omega.pow_u64(i as u64)).collect();

            let z = Ext::from_fp_components(&[
                F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
                F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
            ]).expect("ext from components");

            let by_product = h.iter().fold(Ext::one(), |acc, &w| {
                acc * (z - Ext::from_fp(w))
            });
            let by_formula = vanishing_at_ext::<Ext>(z, t);
            assert_eq!(by_formula, by_product,
                "vanishing form mismatch at T = {}", t);
        }
    }

    /// The check passes when `constraint_at_z := q_at_z · Z_H(z)`.
    /// Witnesses round-trip with synthetic FS challenges.
    #[test]
    fn check_passes_on_consistent_claims() {
        let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
        let trace_len: usize = 16;
        for _ in 0..32 {
            let z = Ext::from_fp_components(&[
                F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
                F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
            ]).expect("ext z");
            let q_at_z = Ext::from_fp(F::rand(&mut rng));
            let z_h = vanishing_at_ext::<Ext>(z, trace_len);
            let constraint_at_z = q_at_z * z_h;

            assert!(check_ood_consistency::<Ext>(z, q_at_z, constraint_at_z, trace_len));
        }
    }

    /// Flipping `q_at_z` produces a non-equal pair w.h.p.
    #[test]
    fn check_rejects_tampered_q_at_z() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
        let trace_len: usize = 16;
        let z = Ext::from_fp_components(&[
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
        ]).expect("ext z");
        let q_at_z = Ext::from_fp(F::rand(&mut rng));
        let z_h = vanishing_at_ext::<Ext>(z, trace_len);
        let constraint_at_z = q_at_z * z_h;

        // Sanity: honest is accepted.
        assert!(check_ood_consistency::<Ext>(z, q_at_z, constraint_at_z, trace_len));

        // Tamper #1: add 1 to q_at_z.
        let tampered_q = q_at_z + Ext::one();
        assert!(!check_ood_consistency::<Ext>(z, tampered_q, constraint_at_z, trace_len));

        // Tamper #2: replace q_at_z with a fresh random scalar.
        let other_q = Ext::from_fp(F::rand(&mut rng));
        assert!(!check_ood_consistency::<Ext>(z, other_q, constraint_at_z, trace_len));
    }

    /// Flipping `constraint_at_z` produces a non-equal pair w.h.p.
    #[test]
    fn check_rejects_tampered_constraint_at_z() {
        let mut rng = StdRng::seed_from_u64(0xBADC_0DE0);
        let trace_len: usize = 8;
        let z = Ext::from_fp_components(&[
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
        ]).expect("ext z");
        let q_at_z = Ext::from_fp(F::rand(&mut rng));
        let z_h = vanishing_at_ext::<Ext>(z, trace_len);
        let constraint_at_z = q_at_z * z_h;

        // Sanity.
        assert!(check_ood_consistency::<Ext>(z, q_at_z, constraint_at_z, trace_len));

        // Tamper: add 1.
        let bad_phi = constraint_at_z + Ext::one();
        assert!(!check_ood_consistency::<Ext>(z, q_at_z, bad_phi, trace_len));

        // Tamper: zero out.
        assert!(!check_ood_consistency::<Ext>(z, q_at_z, Ext::from_fp(F::one()) - Ext::from_fp(F::one()), trace_len));
    }

    /// Wrong trace-length in the verifier ⇒ wrong `Z_H(z)` ⇒ check rejects.
    /// This catches a verifier that mis-aligned its expectation of `H`.
    #[test]
    fn check_rejects_wrong_trace_len() {
        let mut rng = StdRng::seed_from_u64(0x9A9A_9A9A);
        let prover_trace_len: usize = 16;
        let z = Ext::from_fp_components(&[
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
        ]).expect("ext z");
        let q_at_z = Ext::from_fp(F::rand(&mut rng));
        let z_h_prover = vanishing_at_ext::<Ext>(z, prover_trace_len);
        let constraint_at_z = q_at_z * z_h_prover;

        // Honest with matched T.
        assert!(check_ood_consistency::<Ext>(z, q_at_z, constraint_at_z, prover_trace_len));

        // Mismatched T: verifier uses 8 instead of 16.
        let wrong_t: usize = 8;
        assert!(!check_ood_consistency::<Ext>(z, q_at_z, constraint_at_z, wrong_t));

        // And 32 instead of 16.
        let wrong_t2: usize = 32;
        assert!(!check_ood_consistency::<Ext>(z, q_at_z, constraint_at_z, wrong_t2));
    }

    /// `check_ood_consistency_from_claims` wraps the bare primitive.
    /// Exercises the wrapper with a synthetic OodClaims.
    #[test]
    fn from_claims_wrapper_matches_bare_primitive() {
        let mut rng = StdRng::seed_from_u64(0xFADE_FADE);
        let trace_len: usize = 16;
        let w: usize = 2;
        let k_shifts: usize = 2;

        let z = Ext::from_fp_components(&[
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
        ]).expect("ext z");
        let q_at_z = Ext::from_fp(F::rand(&mut rng));
        let trace_at_shifts: Vec<Vec<Ext>> = (0..w)
            .map(|_| (0..k_shifts).map(|_| Ext::from_fp(F::rand(&mut rng))).collect())
            .collect();

        let ood = OodClaims { z, trace_at_shifts, q_at_z };

        let z_h = vanishing_at_ext::<Ext>(z, trace_len);
        let phi = q_at_z * z_h;
        assert!(check_ood_consistency_from_claims::<Ext>(&ood, phi, trace_len));

        // Tampered ood.q_at_z (build a fresh OodClaims with mutated field).
        let bad_ood = OodClaims {
            z: ood.z,
            trace_at_shifts: ood.trace_at_shifts.clone(),
            q_at_z: ood.q_at_z + Ext::one(),
        };
        assert!(!check_ood_consistency_from_claims::<Ext>(&bad_ood, phi, trace_len));
    }

    /// Toy round-trip: a 1-column geometric "AIR" with the recurrence
    /// `T(ω · x) = α · T(x)`.  Constraint at OOD point is
    /// `Φ(z) = T̂(ω z) − α · T̂(z)`.  We construct `q_at_z` such that
    /// `Φ(z) = q_at_z · Z_H(z)` and verify the OOD check accepts.
    ///
    /// This is a synthetic round-trip — it does NOT invoke the full
    /// merge prover; it pins the algebraic shape of how a real AIR
    /// will hook into `check_ood_consistency`.
    #[test]
    fn toy_geometric_air_roundtrips_through_consistency_check() {
        let mut rng = StdRng::seed_from_u64(0x5E47_5247);
        let trace_len: usize = 8;
        let omega_t: F = F::get_root_of_unity(trace_len as u64).expect("two-adic root");

        // Shifts: σ_0 = 1 (current row), σ_1 = ω_T (next row).
        let _shifts: Vec<F> = vec![F::one(), omega_t];

        // Pick OOD point z ∈ Ext.
        let z = Ext::from_fp_components(&[
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
        ]).expect("ext z");

        // Toy AIR ratio α and synthetic OOD trace claims.
        let alpha = Ext::from_fp(F::rand(&mut rng));
        let t_at_z = Ext::from_fp(F::rand(&mut rng));
        let t_at_omega_z = alpha * t_at_z;  // satisfies recurrence at z

        // Φ(z) = T̂(ω z) − α · T̂(z) = 0 (by construction at z).
        // BUT: this is the "AIR holds on H" case; Z_H(z) ≠ 0 generically,
        // so q_at_z = Φ(z) / Z_H(z) = 0 makes the check pass trivially.
        let phi_at_z_zero_air = t_at_omega_z - alpha * t_at_z;
        let z_h = vanishing_at_ext::<Ext>(z, trace_len);
        assert_eq!(phi_at_z_zero_air, Ext::from_fp(F::one()) - Ext::from_fp(F::one()));

        // Trivial-case check: q = 0 ⇒ consistency holds.
        let q_trivial = phi_at_z_zero_air;  // = 0
        assert!(check_ood_consistency::<Ext>(z, q_trivial, phi_at_z_zero_air, trace_len));

        // Non-trivial case: a generic (random) Φ(z); compute matching q.
        let phi_generic = Ext::from_fp(F::rand(&mut rng));
        let z_h_inv = z_h.invert().expect("Z_H(z) is non-zero outside H");
        let q_matched = phi_generic * z_h_inv;
        assert!(check_ood_consistency::<Ext>(z, q_matched, phi_generic, trace_len));

        // Tamper: nudge α, recompute Φ̂(z) honestly, reject.
        let alpha_bad = alpha + Ext::one();
        let phi_bad = t_at_omega_z - alpha_bad * t_at_z;
        assert!(!check_ood_consistency::<Ext>(z, q_matched, phi_bad, trace_len));
    }
}
