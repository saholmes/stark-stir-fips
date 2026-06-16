//! P5.2 — Prover-side layer-0 phase + explicit-form proof envelope.
//!
//! This is the integration glue between the explicit-merge primitives
//! (`Layer0Commit`, `deep_ali_merge_explicit`, `AirOodEvaluator`) and
//! the existing FRI / STIR machinery in `fri.rs`.
//!
//! ## Layer-0 phase (this commit)
//!
//! `prove_layer0_phase` runs the steps the paper describes BEFORE the
//! first FRI fold:
//!
//!   1. Build `Layer0Commit` over `(T_1, …, T_w, Q, R)` on `H_0`.
//!   2. Open the FS transcript, bind the statement, absorb the
//!      layer-0 root (matches fri.rs's `bind_statement_to_transcript`
//!      + root absorb).
//!   3. Draw the OOD point `z ∈ E` (`b"z_fp3"` — same tag as fri.rs).
//!   4. AIR-evaluator → `OodClaims` via
//!      `build_ood_claims_from_witness`.
//!   5. Absorb OOD claims into the transcript.
//!   6. Draw merge-batching challenges γ_1, γ_2, β
//!      (`b"ali_gamma1"`, `b"ali_gamma2"`, `b"ali_beta"`).
//!   7. Compute the merged proximity target `f_0` on `H_0` via
//!      `deep_ali_merge_explicit`.
//!
//! `Layer0PhaseOutput` returns all of (commit, OOD claims, merge
//! challenges, merge output, transcript-state).  The transcript field
//! carries forward to the FRI rounds (P5.3): subsequent layers
//! absorb their commitments into the same Transcript, preserving FS
//! discipline end-to-end.
//!
//! ## Proof envelope (this commit)
//!
//! `DeepFriProofExplicit<E>` and `FriQueryPayloadExplicit<E>` are the
//! wire-format types the explicit prover will produce.  Field layout
//! mirrors `DeepFriProof` from `fri.rs` with two changes:
//!
//!   - `f0_openings` is replaced by `layer0_openings` (a
//!     `Layer0Opening` per query, not a single-element
//!     `MerkleOpening`).
//!   - `ood_claims` is carried explicitly on the wire (the verifier
//!     uses it to recompute `f_0` at queried positions and to run
//!     `check_ood_consistency`).
//!
//! ## Out of scope
//!
//! - FRI rounds 1..L using the merged `f_0_evals_ext` as input — that
//!   requires either a refactor of `fri_build_transcript` or a
//!   parallel "rounds-only" entry point.  Tracked as P5.3.
//! - Verifier `deep_fri_verify_explicit`.  Tracked as P5.4.
//!
//! ## Surgery on fri.rs (visibility only)
//!
//! Four helpers — `safe_field_challenge`, `challenge_ext`,
//! `absorb_ext`, `bind_statement_to_transcript` — were widened from
//! private to `pub(crate)` so this module can share the SAME FS
//! transcript implementation as the existing prover.  Re-implementing
//! these would be a soundness risk (any divergence from fri.rs's FS
//! discipline breaks the joint security argument).

use ark_goldilocks::Goldilocks as F;
use hash::HASH_BYTES;
use transcript::Transcript;

use crate::explicit_merge::{
    deep_ali_merge_explicit, MergeChallenges, MergeOutput, MergeWitness, OodClaims,
};
use crate::explicit_merge_air::{build_ood_claims_from_witness, AirOodEvaluator};
use crate::explicit_merge_layer0::{Layer0Commit, Layer0Opening};
use crate::fri::{
    absorb_ext, bind_statement_to_transcript, challenge_ext, FriLayerProofs,
    LayerOpenPayload, LayerQueryRef, StirProximityPayload,
};
use crate::tower_field::TowerField;

// ═══════════════════════════════════════════════════════════════
//  Phase parameters / output
// ═══════════════════════════════════════════════════════════════

/// Parameters consumed by `prove_layer0_phase`.
///
/// Mirrors the subset of `DeepFriParams` that affects the FS-binding
/// at the layer-0 / OOD boundary.  Subsequent FRI rounds carry
/// independent params (schedule entries are shared).
#[derive(Debug, Clone)]
pub struct Layer0PhaseParams {
    /// FRI fold schedule `[m_0, m_1, …, m_{L-1}]`.  Absorbed into
    /// the statement binding.
    pub schedule: Vec<usize>,

    /// Statement-level seed (mirrors `DeepFriParams::seed_z`; same
    /// tag in the bind).
    pub seed_z: u64,

    /// Whether the final layer is coefficient-committed.  Affects
    /// the bind transcript.
    pub coeff_commit_final: bool,

    /// Whether the prover is in STIR mode.  Affects the bind
    /// transcript and downstream layer wiring.
    pub stir: bool,

    /// Domain-separation tag for the layer-0 Merkle tree.  See
    /// `Layer0Commit::from_witness` doc.
    pub layer0_tree_label: u64,
}

/// What `prove_layer0_phase` returns — all the data the next phase
/// (FRI rounds 1..L) needs to continue.
///
/// Carries `transcript` forward; the FRI prover absorbs each layer's
/// commitment into the SAME transcript instance, preserving FS
/// discipline.
pub struct Layer0PhaseOutput<E: TowerField> {
    /// Layer-0 Merkle commit over `(T_1, …, T_w, Q, R)` on `H_0`.
    /// `Layer0Commit::root` is the wire-format "root_f0" the
    /// verifier sees.
    pub layer0_commit: Layer0Commit,

    /// `(z, T̂(σ_i z), q̂(z))` produced by the AIR-evaluator.
    pub ood_claims: OodClaims<E>,

    /// FS-derived merge-batching challenges drawn AFTER absorbing
    /// the OOD claims.
    pub merge_challenges: MergeChallenges<E>,

    /// Merge output containing `f_0_evals_ext` on `H_0`.  This is
    /// the proximity target fed into FRI layer 1.
    pub merge_output: MergeOutput<E>,

    /// FS transcript state immediately after the merge phase.  Pass
    /// to the FRI-rounds-only entry point in P5.3.
    pub transcript: Transcript,
}

// ═══════════════════════════════════════════════════════════════
//  prove_layer0_phase
// ═══════════════════════════════════════════════════════════════

/// Run the prover's layer-0 phase: commit `(T,Q,R)`, draw the OOD
/// point, build OOD claims via the AIR-evaluator, draw merge
/// challenges, and compute `f_0 = merge(witness; z, ood, γ)` on
/// `H_0`.
///
/// See the file-level doc for the seven-step protocol.
pub fn prove_layer0_phase<E, A>(
    witness: &MergeWitness,
    h0_domain: &[F],
    air: &A,
    params: &Layer0PhaseParams,
) -> Layer0PhaseOutput<E>
where
    E: TowerField,
    A: AirOodEvaluator<E>,
{
    // Preconditions.
    assert!(!witness.trace_columns.is_empty(),
        "prove_layer0_phase: witness has no trace columns");
    assert_eq!(witness.trace_columns[0].len(), h0_domain.len(),
        "prove_layer0_phase: |trace_columns[0]| ({}) must equal |H_0| ({})",
        witness.trace_columns[0].len(), h0_domain.len());
    assert_eq!(air.n_columns(), witness.trace_columns.len(),
        "prove_layer0_phase: AIR n_columns ({}) must equal witness w ({})",
        air.n_columns(), witness.trace_columns.len());
    assert_eq!(air.shifts().len(), witness.k_shifts,
        "prove_layer0_phase: AIR |shifts| ({}) must equal witness k_shifts ({})",
        air.shifts().len(), witness.k_shifts);

    // (i) Layer-0 commit over (T_1, …, T_w, Q, R).
    let layer0_commit = Layer0Commit::from_witness(witness, params.layer0_tree_label);

    // (ii) Open transcript, bind statement, absorb the layer-0 root.
    //      Same hash label as fri.rs (b"FRI/FS").
    let mut tr = Transcript::new_matching_hash(b"FRI/FS");
    bind_statement_to_transcript::<E>(
        &mut tr,
        &params.schedule,
        h0_domain.len(),
        params.seed_z,
        params.coeff_commit_final,
        params.stir,
    );
    tr.absorb_bytes(&layer0_commit.root);

    // (iii) Draw OOD point z ∈ E.  Same tag as fri.rs's z_fp3.
    let z: E = challenge_ext::<E>(&mut tr, b"z_fp3");

    // (iv) AIR → OOD claims via the trait + helper from P5.1.
    let ood: OodClaims<E> = build_ood_claims_from_witness::<E, A>(witness, air, z);

    // (v) Absorb OOD claims into transcript in canonical order:
    //     trace_at_shifts row-major (col, shift_idx), then q_at_z.
    for col in &ood.trace_at_shifts {
        for &v in col {
            absorb_ext::<E>(&mut tr, v);
        }
    }
    absorb_ext::<E>(&mut tr, ood.q_at_z);

    // (vi) Draw merge-batching challenges γ_1, γ_2, β.
    let merge_challenges = MergeChallenges {
        gamma_1: challenge_ext::<E>(&mut tr, b"ali_gamma1"),
        gamma_2: challenge_ext::<E>(&mut tr, b"ali_gamma2"),
        beta:    challenge_ext::<E>(&mut tr, b"ali_beta"),
    };

    // (vii) Compute the merged proximity target on H_0.
    let merge_output = deep_ali_merge_explicit::<E>(
        witness, h0_domain, air.shifts(), &ood, &merge_challenges,
    );

    Layer0PhaseOutput {
        layer0_commit,
        ood_claims: ood,
        merge_challenges,
        merge_output,
        transcript: tr,
    }
}

// ═══════════════════════════════════════════════════════════════
//  Proof envelope
// ═══════════════════════════════════════════════════════════════

/// One opened query for the explicit-form proof.
///
/// Mirrors `fri::FriQueryPayload<E>` but:
///   - `f0_opening: MerkleOpening`       → `layer0_opening: Layer0Opening`
///
/// Layers 1..L payload (`per_layer_refs`, `per_layer_payloads`) is
/// shared with the existing `FriQueryPayload` — the FRI machinery
/// for layers ≥ 1 is identical between the implicit-trace and
/// explicit forms.
#[derive(Clone)]
pub struct FriQueryPayloadExplicit<E: TowerField> {
    pub per_layer_refs:     Vec<LayerQueryRef>,
    pub per_layer_payloads: Vec<LayerOpenPayload<E>>,
    /// Layer-0 opening produced by `Layer0Commit::open`.  Carries
    /// the `(T_1, …, T_w, Q, R)` payload AND the Merkle path; the
    /// verifier reconstructs `f_0` from this via
    /// `Layer0Opening::verify_and_reconstruct`.
    pub layer0_opening: Layer0Opening,
    pub final_index:    usize,
}

/// Explicit-form proof envelope.  Mirrors `fri::DeepFriProof<E>`
/// with the layer-0 swapped to the explicit Layer0Opening form and
/// an explicit `ood_claims` field on the wire.
///
/// NOT constructed by `prove_layer0_phase`; this is the data shape
/// the P5.3 prover entry point will produce.
pub struct DeepFriProofExplicit<E: TowerField> {
    /// = `Layer0Commit::root` on the prover side.
    pub root_f0: [u8; HASH_BYTES],

    /// Per-layer Merkle roots for layers 1..L (same as
    /// `DeepFriProof::roots`).
    pub roots: Vec<[u8; HASH_BYTES]>,

    /// OOD claims `(z, T̂(σ_i z), q̂(z))` on the wire.
    /// The verifier:
    ///   1. Absorbs these into its FS replay.
    ///   2. Reconstructs `f_0` at queried positions via
    ///      `Layer0Opening::verify_and_reconstruct`.
    ///   3. Runs `check_ood_consistency` with `constraint_at_z`
    ///      from its own `AirOodEvaluator`.
    pub ood_claims: OodClaims<E>,

    /// FRI per-layer Merkle proofs (same as `DeepFriProof`).
    pub layer_proofs: FriLayerProofs,

    /// Per-query opened payloads (explicit-form, see
    /// `FriQueryPayloadExplicit`).
    pub queries: Vec<FriQueryPayloadExplicit<E>>,

    /// `f_ℓ(z_ℓ)` per layer (same as `DeepFriProof::fz_per_layer`).
    pub fz_per_layer: Vec<E>,

    /// Final-layer polynomial coefficients (same as
    /// `DeepFriProof::final_poly_coeffs`).
    pub final_poly_coeffs: Vec<E>,

    pub n0:     usize,
    pub omega0: F,

    /// STIR-specific: coefficient-commit final layer plumbing.
    pub coeff_tuples: Option<Vec<Vec<E>>>,
    pub coeff_root:   Option<[u8; HASH_BYTES]>,

    /// STIR-specific: proximity-query data + coset evals.
    pub stir_coset_evals: Option<Vec<Vec<E>>>,
    pub stir_proximity_queries: Option<Vec<StirProximityPayload<E>>>,

    /// Explicit-form metadata the verifier needs to rebuild the
    /// Merkle config independently.
    pub layer0_tree_label: u64,
    pub trace_width:       usize,
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit_merge_ood_check::{check_ood_consistency_from_claims, vanishing_at_ext};
    use crate::sextic_ext::SexticExt;
    use ark_ff::{FftField, Field, One, UniformRand, Zero};
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use rand::{rngs::StdRng, SeedableRng};

    type Ext = SexticExt;

    // ── Toy AIR re-stated here so this module is self-contained.

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

        // Random Q of degree ≤ T-1.
        let q_t_evals: Vec<F> = (0..trace_len).map(|_| F::rand(rng)).collect();
        let q_domain = GeneralEvaluationDomain::<F>::new(trace_len).unwrap();
        let q_coeffs: Vec<F> = q_domain.ifft(&q_t_evals);
        let q_on_h0: Vec<F> = h0.iter().map(|&x| {
            let mut acc = F::zero();
            for &cc in q_coeffs.iter().rev() { acc = acc * x + cc; }
            acc
        }).collect();

        // T(x) = Q(x) · Z_H(x) + c.
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

    fn baseline_params() -> Layer0PhaseParams {
        Layer0PhaseParams {
            schedule: vec![2, 2, 2],
            seed_z: 0xC0FFEE,
            coeff_commit_final: false,
            stir: false,
            layer0_tree_label: 0x52_0001,
        }
    }

    // ── Honest e2e ──

    /// `prove_layer0_phase` on an honest constant-boundary witness:
    ///   - returns a populated commit + non-zero ood + non-zero f_0,
    ///   - OOD consistency check accepts.
    #[test]
    fn prove_layer0_phase_honest_e2e_passes_ood_check() {
        let mut rng = StdRng::seed_from_u64(0x5201_0001);
        let (witness, h0) = build_honest_witness(
            &mut rng, 8, 4, F::from(13u64));
        let air = ConstantBoundaryAir { c: F::from(13u64) };
        let params = baseline_params();

        let out: Layer0PhaseOutput<Ext> =
            prove_layer0_phase::<Ext, _>(&witness, &h0, &air, &params);

        // Commit root non-zero.
        assert_ne!(out.layer0_commit.root, [0u8; HASH_BYTES]);

        // f_0 not trivially zero.
        let zero = Ext::zero();
        let nz = out.merge_output.f0_evals_ext.iter().filter(|&&v| v != zero).count();
        assert!(nz > 0, "f_0 should not be trivially zero");

        // OOD consistency.
        let phi = air.constraint_at_z(out.ood_claims.z, &out.ood_claims.trace_at_shifts);
        let z_h = vanishing_at_ext::<Ext>(out.ood_claims.z, witness.trace_len);
        assert_eq!(phi, out.ood_claims.q_at_z * z_h,
            "honest constant-boundary witness violates Φ = Q · Z_H");
        assert!(check_ood_consistency_from_claims::<Ext>(
            &out.ood_claims, phi, witness.trace_len,
        ), "honest witness must pass OOD consistency");
    }

    // ── Determinism ──

    /// Same inputs → same root, same OOD point, same merge
    /// challenges, same f_0 — i.e. no hidden RNG / clock / nonce
    /// snuck into the prover path.
    #[test]
    fn prove_layer0_phase_deterministic_in_public_inputs() {
        let mut rng = StdRng::seed_from_u64(0x5202_0002);
        let (witness, h0) = build_honest_witness(
            &mut rng, 8, 4, F::from(5u64));
        let air = ConstantBoundaryAir { c: F::from(5u64) };
        let params = baseline_params();

        let a: Layer0PhaseOutput<Ext> =
            prove_layer0_phase::<Ext, _>(&witness, &h0, &air, &params);
        let b: Layer0PhaseOutput<Ext> =
            prove_layer0_phase::<Ext, _>(&witness, &h0, &air, &params);

        assert_eq!(a.layer0_commit.root, b.layer0_commit.root);
        assert_eq!(a.ood_claims.z, b.ood_claims.z);
        assert_eq!(a.ood_claims.q_at_z, b.ood_claims.q_at_z);
        assert_eq!(a.merge_challenges.gamma_1, b.merge_challenges.gamma_1);
        assert_eq!(a.merge_challenges.gamma_2, b.merge_challenges.gamma_2);
        assert_eq!(a.merge_challenges.beta,    b.merge_challenges.beta);
        assert_eq!(a.merge_output.f0_evals_ext, b.merge_output.f0_evals_ext);
    }

    // ── Tamper grid ──

    /// Tampering the witness AFTER calling prove_layer0_phase has no
    /// effect on the original output (defence against re-use bugs).
    /// And re-running prove on the tampered witness produces a
    /// different root.
    #[test]
    fn tamper_witness_changes_layer0_root() {
        let mut rng = StdRng::seed_from_u64(0x5203_0003);
        let (witness, h0) = build_honest_witness(
            &mut rng, 8, 4, F::from(9u64));
        let air = ConstantBoundaryAir { c: F::from(9u64) };
        let params = baseline_params();

        let orig = prove_layer0_phase::<Ext, _>(&witness, &h0, &air, &params);

        let mut tampered = witness.clone();
        tampered.trace_columns[0][2] += F::one();
        let tampered_out = prove_layer0_phase::<Ext, _>(&tampered, &h0, &air, &params);

        assert_ne!(orig.layer0_commit.root, tampered_out.layer0_commit.root,
            "tampering T_0[2] must change the layer-0 root");
        // And — because the layer-0 root changes — z changes too
        // (z is FS-derived from the root).
        assert_ne!(orig.ood_claims.z, tampered_out.ood_claims.z,
            "tampering T_0[2] must change the FS-derived OOD point z");
    }

    /// Different `layer0_tree_label` ⇒ different root ⇒ different z.
    #[test]
    fn different_tree_label_changes_layer0_root_and_z() {
        let mut rng = StdRng::seed_from_u64(0x5204_0004);
        let (witness, h0) = build_honest_witness(
            &mut rng, 8, 4, F::from(2u64));
        let air = ConstantBoundaryAir { c: F::from(2u64) };

        let p_a = Layer0PhaseParams {
            layer0_tree_label: 0xAAAA_AAAA,
            ..baseline_params()
        };
        let p_b = Layer0PhaseParams {
            layer0_tree_label: 0xBBBB_BBBB,
            ..baseline_params()
        };

        let a = prove_layer0_phase::<Ext, _>(&witness, &h0, &air, &p_a);
        let b = prove_layer0_phase::<Ext, _>(&witness, &h0, &air, &p_b);

        assert_ne!(a.layer0_commit.root, b.layer0_commit.root);
        assert_ne!(a.ood_claims.z, b.ood_claims.z);
    }

    /// Different `seed_z` ⇒ same root but different z (statement bind
    /// changes; layer-0 commit is independent of seed_z).
    #[test]
    fn different_seed_z_keeps_root_but_changes_z() {
        let mut rng = StdRng::seed_from_u64(0x5205_0005);
        let (witness, h0) = build_honest_witness(
            &mut rng, 8, 4, F::from(4u64));
        let air = ConstantBoundaryAir { c: F::from(4u64) };

        let p_a = Layer0PhaseParams { seed_z: 0x1111_1111, ..baseline_params() };
        let p_b = Layer0PhaseParams { seed_z: 0x2222_2222, ..baseline_params() };

        let a = prove_layer0_phase::<Ext, _>(&witness, &h0, &air, &p_a);
        let b = prove_layer0_phase::<Ext, _>(&witness, &h0, &air, &p_b);

        assert_eq!(a.layer0_commit.root, b.layer0_commit.root,
            "layer-0 commit must be independent of statement-level seed_z");
        assert_ne!(a.ood_claims.z, b.ood_claims.z,
            "z is FS-derived through bind_statement_to_transcript → seed_z must affect z");
    }

    /// Different schedule ⇒ same root but different z (schedule
    /// participates in the statement bind).
    #[test]
    fn different_schedule_keeps_root_but_changes_z() {
        let mut rng = StdRng::seed_from_u64(0x5206_0006);
        let (witness, h0) = build_honest_witness(
            &mut rng, 8, 4, F::from(8u64));
        let air = ConstantBoundaryAir { c: F::from(8u64) };

        let p_a = Layer0PhaseParams { schedule: vec![2, 2, 2], ..baseline_params() };
        let p_b = Layer0PhaseParams { schedule: vec![2, 2, 4], ..baseline_params() };

        let a = prove_layer0_phase::<Ext, _>(&witness, &h0, &air, &p_a);
        let b = prove_layer0_phase::<Ext, _>(&witness, &h0, &air, &p_b);

        assert_eq!(a.layer0_commit.root, b.layer0_commit.root);
        assert_ne!(a.ood_claims.z, b.ood_claims.z);
    }

    /// AIR shift-count mismatch with witness.k_shifts panics with a
    /// clear message.
    #[test]
    #[should_panic(expected = "AIR |shifts|")]
    fn shift_count_mismatch_panics() {
        let mut rng = StdRng::seed_from_u64(0x5207_0007);
        let (mut witness, h0) = build_honest_witness(
            &mut rng, 8, 4, F::from(3u64));
        witness.k_shifts = 2;  // AIR uses 1 shift
        let air = ConstantBoundaryAir { c: F::from(3u64) };
        let params = baseline_params();
        let _ = prove_layer0_phase::<Ext, _>(&witness, &h0, &air, &params);
    }

    /// `DeepFriProofExplicit` and `FriQueryPayloadExplicit` compile
    /// and are constructible from default-shaped pieces (smoke test
    /// on the wire-format types — no semantics tested yet, that's
    /// P5.3).
    #[test]
    fn proof_envelope_types_construct_under_minimal_input() {
        use crate::explicit_merge::Layer0LeafContent;
        let leaf = Layer0LeafContent {
            trace_values: vec![F::one()],
            q_value: F::one(),
            r_value: F::one(),
        };
        let label = 0xDEAD_DEAD;
        let commit = Layer0Commit::from_leaves(&[leaf.clone()], label);
        let opening = commit.open(0);

        let q = FriQueryPayloadExplicit::<Ext> {
            per_layer_refs:     vec![],
            per_layer_payloads: vec![],
            layer0_opening:     opening,
            final_index:        0,
        };

        let proof = DeepFriProofExplicit::<Ext> {
            root_f0: commit.root,
            roots:   vec![],
            ood_claims: OodClaims {
                z: Ext::one(),
                trace_at_shifts: vec![vec![Ext::one()]],
                q_at_z: Ext::one(),
            },
            layer_proofs: FriLayerProofs { layers: vec![] },
            queries: vec![q],
            fz_per_layer: vec![],
            final_poly_coeffs: vec![],
            n0: 1,
            omega0: F::one(),
            coeff_tuples: None,
            coeff_root: None,
            stir_coset_evals: None,
            stir_proximity_queries: None,
            layer0_tree_label: label,
            trace_width: 1,
        };

        assert_eq!(proof.queries.len(), 1);
        assert_eq!(proof.layer0_tree_label, label);
        assert_eq!(proof.trace_width, 1);
    }
}
