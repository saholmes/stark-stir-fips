//! P5.4 — Verifier-side layer-0 phase.
//!
//! Mirrors `prove_layer0_phase` (P5.2) on the verifier side: replays
//! the FS transcript over `(statement bind ∥ root_f0 ∥ OOD claims)`
//! to re-derive the OOD point `z` and the merge challenges γ_1/γ_2/β,
//! and runs the OOD consistency check (P3 `check_ood_consistency`)
//! via the AIR-evaluator.
//!
//! ## FS replay
//!
//! Verifier and prover agree on the same Transcript construction:
//!
//!   1. `Transcript::new_matching_hash(b"FRI/FS")`
//!   2. `bind_statement_to_transcript` with the same parameters.
//!   3. Absorb `root_f0` (verifier reads from the proof).
//!   4. Draw `z = challenge_ext(b"z_fp3")` and check it matches
//!      `ood_claims.z` (z_matches gate).
//!   5. Absorb `ood_claims.trace_at_shifts` (row-major) then
//!      `ood_claims.q_at_z`.
//!   6. Draw `γ_1 = challenge_ext(b"ali_gamma1")`,
//!      `γ_2 = challenge_ext(b"ali_gamma2")`,
//!      `β   = challenge_ext(b"ali_beta")`.
//!   7. Compute `Φ(z) = air.constraint_at_z(z, trace_at_shifts)` and
//!      check `Φ(z) == q_at_z · Z_H(z)` (ood_consistent gate).
//!
//! The returned transcript state is post-merge-challenge-draw,
//! exactly aligned with the prover's transcript at the same point;
//! the FRI-rounds-1..L verifier (P5.5) absorbs each layer
//! commitment into this transcript to keep the FS chain unbroken.
//!
//! ## What is and isn't checked at this layer
//!
//! - `z_matches` catches: `root_f0` tamper, `ood_claims.z` tamper,
//!   any change to a parameter absorbed in `bind_statement_to_transcript`
//!   (schedule, seed_z, n0, coeff_commit_final, stir).
//!
//! - `ood_consistent` catches: `ood_claims.q_at_z` tamper,
//!   `ood_claims.trace_at_shifts` tamper, and any AIR-evaluator
//!   mismatch between prover and verifier.
//!
//! - This layer does NOT verify Layer-0 Merkle openings or
//!   FRI-rounds Merkle openings — those happen at query time (P5.5).

use ark_goldilocks::Goldilocks as F;
use hash::HASH_BYTES;
use transcript::Transcript;

use crate::explicit_merge::{MergeChallenges, OodClaims};
use crate::explicit_merge_air::AirOodEvaluator;
use crate::explicit_merge_ood_check::check_ood_consistency_from_claims;
use crate::explicit_merge_prove::Layer0PhaseParams;
use crate::fri::{absorb_ext, bind_statement_to_transcript, challenge_ext};
use crate::tower_field::TowerField;

// ─── Verifier-side output ────────────────────────────────────────

/// What `verify_layer0_phase` returns.
///
/// Both `z_matches` AND `ood_consistent` must be true for the
/// verifier to accept the layer-0 phase.  The `is_accepted()` helper
/// folds both into a single decision; granular flags are surfaced so
/// callers can produce richer diagnostics in test logs.
pub struct Layer0VerifyOutput<E: TowerField> {
    /// The OOD point the verifier re-derived from the Transcript.
    /// Compared against `ood_claims.z` for the `z_matches` gate.
    pub redrawn_z: E,

    /// Merge FS challenges (γ_1, γ_2, β) re-drawn from the
    /// Transcript after absorbing OOD claims.  Identical to the
    /// prover's `Layer0PhaseOutput::merge_challenges` on an honest
    /// transcript.
    pub merge_challenges: MergeChallenges<E>,

    /// Transcript state immediately after the merge-challenge draws.
    /// Pass to the FRI-rounds-1..L verifier (P5.5).
    pub transcript: Transcript,

    /// `(redrawn_z == ood_claims.z)`.  Closes any FS-binding tamper
    /// upstream of the OOD point draw.
    pub z_matches: bool,

    /// `check_ood_consistency_from_claims(ood_claims, Φ(z), trace_len)`.
    /// Closes any tamper of `q_at_z`, `trace_at_shifts`, or the AIR
    /// evaluator.
    pub ood_consistent: bool,
}

impl<E: TowerField> Layer0VerifyOutput<E> {
    /// Whether the layer-0 phase is accepted.  Both gates must pass.
    pub fn is_accepted(&self) -> bool {
        self.z_matches && self.ood_consistent
    }
}

// ─── verify_layer0_phase ─────────────────────────────────────────

/// Verifier-side layer-0 phase.
///
/// See the file-level doc for the FS replay sequence.
///
/// Inputs:
///   - `root_f0`: from the proof.  In the explicit form this is
///     `Layer0Commit::root`; in the classic form it's the single-
///     element-leaf f_0 Merkle root.  The verifier doesn't need to
///     distinguish at this layer — both flows absorb the same field.
///   - `n0`: size of `H_0`, from the proof.  Participates in the
///     statement bind.
///   - `ood_claims`: from the proof.
///   - `air`: the AIR-evaluator (must match the prover's).
///   - `trace_len`: trace-domain size `T`, from the AIR / from the
///     proof if carried separately.
///   - `params`: same fields as on the prover side; `layer0_tree_label`
///     is unused at this layer (it enters Merkle-path verification
///     in P5.5).
pub fn verify_layer0_phase<E, A>(
    root_f0: [u8; HASH_BYTES],
    n0: usize,
    ood_claims: &OodClaims<E>,
    air: &A,
    trace_len: usize,
    params: &Layer0PhaseParams,
) -> Layer0VerifyOutput<E>
where
    E: TowerField,
    A: AirOodEvaluator<E>,
{
    // (i) Open transcript and bind statement.
    let mut tr = Transcript::new_matching_hash(b"FRI/FS");
    bind_statement_to_transcript::<E>(
        &mut tr,
        &params.schedule,
        n0,
        params.seed_z,
        params.coeff_commit_final,
        params.stir,
    );

    // (ii) Absorb the (claimed) layer-0 root from the proof.
    tr.absorb_bytes(&root_f0);

    // (iii) Re-derive z; compare to the claimed z.
    let redrawn_z: E = challenge_ext::<E>(&mut tr, b"z_fp3");
    let z_matches = redrawn_z == ood_claims.z;

    // (iv) Absorb OOD claims in the same order as the prover.
    for col in &ood_claims.trace_at_shifts {
        for &v in col {
            absorb_ext::<E>(&mut tr, v);
        }
    }
    absorb_ext::<E>(&mut tr, ood_claims.q_at_z);

    // (v) Re-derive merge FS challenges.
    let merge_challenges = MergeChallenges {
        gamma_1: challenge_ext::<E>(&mut tr, b"ali_gamma1"),
        gamma_2: challenge_ext::<E>(&mut tr, b"ali_gamma2"),
        beta:    challenge_ext::<E>(&mut tr, b"ali_beta"),
    };

    // (vi) OOD consistency.  Compute Φ̂(z) from the OOD trace claims
    //      via the AIR-evaluator, then check Φ̂(z) =? q̂(z) · Z_H(z).
    let constraint_at_z = air.constraint_at_z(ood_claims.z, &ood_claims.trace_at_shifts);
    let ood_consistent = check_ood_consistency_from_claims::<E>(
        ood_claims, constraint_at_z, trace_len,
    );

    Layer0VerifyOutput {
        redrawn_z,
        merge_challenges,
        transcript: tr,
        z_matches,
        ood_consistent,
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit_merge::MergeWitness;
    use crate::explicit_merge_prove::{prove_explicit_state, ExplicitProverState};
    use crate::fri::{FriDomain, FriProverParams};
    use crate::sextic_ext::SexticExt;
    use ark_ff::{FftField, Field, One, UniformRand, Zero};
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use rand::{rngs::StdRng, SeedableRng};

    type Ext = SexticExt;

    // Re-state toy AIR (mirror of explicit_merge_prove::tests).

    struct ConstantBoundaryAir { c: F }

    impl AirOodEvaluator<Ext> for ConstantBoundaryAir {
        fn n_columns(&self) -> usize { 1 }
        fn shifts(&self) -> &[F] {
            const ONES: [F; 1] = [F::ONE];
            &ONES
        }
        fn constraint_at_z(&self, _z: Ext, trace_at_shifts: &[Vec<Ext>]) -> Ext {
            trace_at_shifts[0][0] - Ext::from_fp(self.c)
        }
    }

    fn build_honest_witness(
        rng: &mut StdRng, trace_len: usize, blowup: usize, c: F,
    ) -> (MergeWitness, Vec<F>) {
        let n = trace_len * blowup;
        let omega_n: F = F::get_root_of_unity(n as u64).expect("two-adic root");
        let h0: Vec<F> = (0..n).map(|i| omega_n.pow_u64(i as u64)).collect();

        let q_t_evals: Vec<F> = (0..trace_len).map(|_| F::rand(rng)).collect();
        let q_domain = GeneralEvaluationDomain::<F>::new(trace_len).unwrap();
        let q_coeffs: Vec<F> = q_domain.ifft(&q_t_evals);
        let q_on_h0: Vec<F> = h0.iter().map(|&x| {
            let mut acc = F::zero();
            for &cc in q_coeffs.iter().rev() { acc = acc * x + cc; }
            acc
        }).collect();

        let t_on_h0: Vec<F> = h0.iter().zip(q_on_h0.iter())
            .map(|(&x, &q)| q * (x.pow([trace_len as u64]) - F::one()) + c)
            .collect();

        let r: Vec<F> = (0..n).map(|_| F::rand(rng)).collect();

        let d_c = 2usize;
        let d0 = (d_c - 1) * trace_len - 1;
        let w = MergeWitness {
            trace_columns: vec![t_on_h0],
            ali_quotient: q_on_h0,
            blinder: r,
            trace_len, d0, k_shifts: 1,
        };
        (w, h0)
    }

    fn baseline_fri_params() -> FriProverParams {
        FriProverParams {
            schedule: vec![2, 2, 2],
            seed_z: 0xC0FFEE,
            coeff_commit_final: false,
            d_final: 1,
            stir: false,
        }
    }

    fn baseline_layer0_params() -> Layer0PhaseParams {
        Layer0PhaseParams {
            schedule: vec![2, 2, 2],
            seed_z: 0xC0FFEE,
            coeff_commit_final: false,
            stir: false,
            layer0_tree_label: 0x54_AAAA,
        }
    }

    fn run_honest_prover(
        seed: u64, c_val: u64,
    ) -> (ExplicitProverState<Ext>, MergeWitness, Vec<F>, usize, ConstantBoundaryAir, FriProverParams) {
        let mut rng = StdRng::seed_from_u64(seed);
        let trace_len = 8usize;
        let blowup = 4usize;
        let n = trace_len * blowup;
        let c = F::from(c_val);
        let (witness, h0) = build_honest_witness(&mut rng, trace_len, blowup, c);
        let air = ConstantBoundaryAir { c };
        let fri_params = baseline_fri_params();
        let domain0 = FriDomain::new_radix2(n);
        let label = baseline_layer0_params().layer0_tree_label;
        let state = prove_explicit_state::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, label);
        (state, witness, h0, trace_len, air, fri_params)
    }

    // ── Honest round-trip ──

    /// Honest prover ↔ verifier: z_matches AND ood_consistent both
    /// true; redrawn merge-challenges match prover's.
    #[test]
    fn verify_layer0_phase_accepts_honest_prover() {
        let (state, _w, _h0, trace_len, air, _fp) =
            run_honest_prover(0x5401_0001, 13);

        let layer0_params = baseline_layer0_params();
        let out: Layer0VerifyOutput<Ext> = verify_layer0_phase::<Ext, _>(
            state.layer0_commit.root,
            state.fri_state.f_layers_ext[0].len(),  // n0
            &state.ood_claims,
            &air,
            trace_len,
            &layer0_params,
        );

        assert!(out.is_accepted(), "honest prover must be accepted");
        assert!(out.z_matches);
        assert!(out.ood_consistent);
        assert_eq!(out.redrawn_z, state.ood_claims.z);
        assert_eq!(out.merge_challenges.gamma_1, state.merge_challenges.gamma_1);
        assert_eq!(out.merge_challenges.gamma_2, state.merge_challenges.gamma_2);
        assert_eq!(out.merge_challenges.beta,    state.merge_challenges.beta);
    }

    // ── Tamper: ood_claims.q_at_z ──

    /// Tampering `q_at_z` leaves the z draw intact (z is FS-derived
    /// BEFORE q_at_z is absorbed) but breaks the OOD consistency
    /// equation.  `z_matches` true, `ood_consistent` false.
    #[test]
    fn tamper_q_at_z_keeps_z_match_but_breaks_ood_consistency() {
        let (state, _w, _h0, trace_len, air, _fp) =
            run_honest_prover(0x5402_0002, 11);

        let mut ood_bad = state.ood_claims.clone();
        ood_bad.q_at_z += Ext::one();

        let out = verify_layer0_phase::<Ext, _>(
            state.layer0_commit.root,
            state.fri_state.f_layers_ext[0].len(),
            &ood_bad, &air, trace_len,
            &baseline_layer0_params(),
        );

        assert!(out.z_matches, "tampering q_at_z does NOT affect z draw");
        assert!(!out.ood_consistent, "tampering q_at_z must break OOD consistency");
        assert!(!out.is_accepted());
    }

    // ── Tamper: ood_claims.trace_at_shifts ──

    /// Tampering a `trace_at_shifts` entry changes Φ(z) the AIR
    /// computes, so the OOD consistency equation breaks.  Note:
    /// trace_at_shifts is absorbed BEFORE q_at_z, so the merge
    /// challenges γ_1/γ_2/β diverge too — but at this layer we only
    /// surface z_matches and ood_consistent.
    #[test]
    fn tamper_trace_at_shifts_breaks_ood_consistency() {
        let (state, _w, _h0, trace_len, air, _fp) =
            run_honest_prover(0x5403_0003, 9);

        let mut ood_bad = state.ood_claims.clone();
        ood_bad.trace_at_shifts[0][0] += Ext::one();

        let out = verify_layer0_phase::<Ext, _>(
            state.layer0_commit.root,
            state.fri_state.f_layers_ext[0].len(),
            &ood_bad, &air, trace_len,
            &baseline_layer0_params(),
        );

        assert!(out.z_matches, "tampering trace_at_shifts does NOT affect z draw");
        assert!(!out.ood_consistent, "tampering trace_at_shifts must break OOD consistency");
        assert!(!out.is_accepted());
    }

    // ── Tamper: ood_claims.z field ──

    /// Tampering the `z` field of OodClaims (the proof's claim of
    /// what z is) is detected by `z_matches`.  The verifier's
    /// redrawn z is unchanged (the transcript bind & root_f0 are
    /// honest), but the claim differs.
    #[test]
    fn tamper_ood_z_claim_breaks_z_match() {
        let (state, _w, _h0, trace_len, air, _fp) =
            run_honest_prover(0x5404_0004, 5);

        let mut ood_bad = state.ood_claims.clone();
        ood_bad.z += Ext::one();

        let out = verify_layer0_phase::<Ext, _>(
            state.layer0_commit.root,
            state.fri_state.f_layers_ext[0].len(),
            &ood_bad, &air, trace_len,
            &baseline_layer0_params(),
        );

        assert!(!out.z_matches, "z_matches must reject a tampered ood_claims.z");
        // Note: ood_consistent runs against ood_bad.z too (the
        // verifier honours the proof's z field for the consistency
        // check; the z_matches gate is the actual binding gate).
        // We don't assert on ood_consistent here.
        assert!(!out.is_accepted());
    }

    // ── Tamper: root_f0 ──

    /// Tampering the (claimed) root_f0 changes the FS-derived z;
    /// the verifier rejects via z_matches.
    #[test]
    fn tamper_root_f0_breaks_z_match() {
        let (state, _w, _h0, trace_len, air, _fp) =
            run_honest_prover(0x5405_0005, 17);

        let mut bad_root = state.layer0_commit.root;
        bad_root[0] ^= 1;

        let out = verify_layer0_phase::<Ext, _>(
            bad_root,
            state.fri_state.f_layers_ext[0].len(),
            &state.ood_claims, &air, trace_len,
            &baseline_layer0_params(),
        );

        assert!(!out.z_matches, "tampering root_f0 must break z_match");
        assert!(!out.is_accepted());
    }

    // ── Tamper: bind-statement params ──

    /// Tampering schedule on the verifier side ⇒ bind absorbs
    /// different bytes ⇒ different z draw ⇒ z_matches false.
    #[test]
    fn verifier_param_mismatch_breaks_z_match() {
        let (state, _w, _h0, trace_len, air, _fp) =
            run_honest_prover(0x5406_0006, 21);

        let mut bad_params = baseline_layer0_params();
        bad_params.schedule = vec![2, 4, 2];  // different from prover

        let out = verify_layer0_phase::<Ext, _>(
            state.layer0_commit.root,
            state.fri_state.f_layers_ext[0].len(),
            &state.ood_claims, &air, trace_len,
            &bad_params,
        );

        assert!(!out.z_matches);
        assert!(!out.is_accepted());
    }

    // ── Tamper: trace_len mismatch ──

    /// Verifier using the wrong trace_len computes a different
    /// Z_H(z) and fails ood_consistent.
    #[test]
    fn verifier_wrong_trace_len_breaks_ood_consistency() {
        let (state, _w, _h0, _trace_len, air, _fp) =
            run_honest_prover(0x5407_0007, 3);

        // Prover used trace_len = 8; verifier uses 16.
        let out = verify_layer0_phase::<Ext, _>(
            state.layer0_commit.root,
            state.fri_state.f_layers_ext[0].len(),
            &state.ood_claims, &air, /* wrong */ 16,
            &baseline_layer0_params(),
        );

        assert!(out.z_matches);
        assert!(!out.ood_consistent);
        assert!(!out.is_accepted());
    }

    // ── Determinism ──

    /// Same inputs → identical Layer0VerifyOutput (excluding
    /// transcript, which has no stable equality).  Pinned via
    /// redrawn_z + merge_challenges + flags.
    #[test]
    fn verify_layer0_phase_deterministic() {
        let (state, _w, _h0, trace_len, air, _fp) =
            run_honest_prover(0x5408_0008, 23);

        let a = verify_layer0_phase::<Ext, _>(
            state.layer0_commit.root,
            state.fri_state.f_layers_ext[0].len(),
            &state.ood_claims, &air, trace_len,
            &baseline_layer0_params(),
        );
        let b = verify_layer0_phase::<Ext, _>(
            state.layer0_commit.root,
            state.fri_state.f_layers_ext[0].len(),
            &state.ood_claims, &air, trace_len,
            &baseline_layer0_params(),
        );

        assert_eq!(a.redrawn_z, b.redrawn_z);
        assert_eq!(a.merge_challenges.gamma_1, b.merge_challenges.gamma_1);
        assert_eq!(a.merge_challenges.gamma_2, b.merge_challenges.gamma_2);
        assert_eq!(a.merge_challenges.beta,    b.merge_challenges.beta);
        assert_eq!(a.z_matches, b.z_matches);
        assert_eq!(a.ood_consistent, b.ood_consistent);
    }
}
