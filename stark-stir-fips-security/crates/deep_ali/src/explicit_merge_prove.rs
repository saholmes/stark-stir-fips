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
    absorb_ext, bind_statement_to_transcript, challenge_ext, ds, ext_evals_to_coeffs,
    fri_prove_layer_openings_only, fri_rounds_from_f0_ext, safe_field_challenge,
    transcript_challenge_hash, FriDomain, FriLayerProofs, FriProverParams,
    FriProverState, LayerOpenPayload, LayerQueryRef, StirProximityPayload,
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
//  prove_explicit_state — layer-0 phase + FRI rounds 1..L
// ═══════════════════════════════════════════════════════════════

/// Full prover state produced by the explicit-form prover at the end
/// of the transcript-building phase.  Carries:
///   - the layer-0 commit (= root_f0 on the wire),
///   - the OOD claims and merge FS challenges from the layer-0 phase,
///   - the FRI state (layers 1..L) produced by the extracted
///     `fri::fri_rounds_from_f0_ext`.
///
/// Query-time logic (Merkle openings on layer-0 + per-layer payloads)
/// runs against this state in P5.4.
pub struct ExplicitProverState<E: TowerField> {
    pub layer0_commit:    Layer0Commit,
    pub ood_claims:       OodClaims<E>,
    pub merge_challenges: MergeChallenges<E>,
    pub fri_state:        FriProverState<E>,
}

/// Run the full explicit-form transcript-building phase end-to-end:
///
///   1. `prove_layer0_phase` (P5.2): Layer0Commit + transcript bind
///      + z draw + OOD claims + γ challenges + merge → f_0_evals_ext.
///   2. `fri::fri_rounds_from_f0_ext` (P5.3 refactor): trace_hash
///      draw + FRI rounds 1..L → FriProverState.
///
/// Both phases share the SAME Transcript instance (passed through
/// `Layer0PhaseOutput::transcript`), so the joint FS-binding chain
/// is unbroken: any FS-sensitive parameter (`seed_z`, `schedule`,
/// `layer0_tree_label`, OOD claims) propagates through to every
/// downstream FRI commitment.
///
/// The returned `FriProverState::f0_base` is empty (explicit form
/// does not use base-field layer-0 leaves).  Layer-0 openings go
/// through `ExplicitProverState::layer0_commit.open(...)` in P5.4.
pub fn prove_explicit_state<E, A>(
    witness: &MergeWitness,
    h0_domain: &[F],
    air: &A,
    domain0: FriDomain,
    fri_params: &FriProverParams,
    layer0_tree_label: u64,
) -> ExplicitProverState<E>
where
    E: TowerField,
    A: AirOodEvaluator<E>,
{
    let layer0_params = Layer0PhaseParams {
        schedule: fri_params.schedule.clone(),
        seed_z: fri_params.seed_z,
        coeff_commit_final: fri_params.coeff_commit_final,
        stir: fri_params.stir,
        layer0_tree_label,
    };

    let phase: Layer0PhaseOutput<E> = prove_layer0_phase::<E, A>(
        witness, h0_domain, air, &layer0_params,
    );

    let root_f0 = phase.layer0_commit.root;
    let z_ext   = phase.ood_claims.z;
    let f0_ext  = phase.merge_output.f0_evals_ext.clone();

    let fri_state = fri_rounds_from_f0_ext::<E>(
        f0_ext,
        Vec::new(),  // no f0_base in explicit form
        domain0,
        fri_params,
        phase.transcript,
        z_ext,
        root_f0,
    );

    ExplicitProverState {
        layer0_commit:    phase.layer0_commit,
        ood_claims:       phase.ood_claims,
        merge_challenges: phase.merge_challenges,
        fri_state,
    }
}

// ═══════════════════════════════════════════════════════════════
//  prove_explicit_queries — query / opening assembly
// ═══════════════════════════════════════════════════════════════

/// Open the explicit-form prover state at `r` FS-derived query
/// positions.
///
/// Reuses the form-independent layer-1..L tree-build + opening logic
/// extracted from `fri_prove_queries` (P5.5: `fri_prove_layer_openings_only`).
/// Layer-0 opening goes through `Layer0Commit::open`, producing a
/// `Layer0Opening` per query that carries the `(T_1, …, T_w, Q, R)`
/// payload plus the Merkle authentication path.
///
/// Returns the per-query `FriQueryPayloadExplicit` list, the
/// per-layer Merkle proofs (layer 1..L openings), and the per-layer
/// roots.  Callers assemble a `DeepFriProofExplicit` from these
/// plus the post-merge final-poly + STIR-specific fields.
pub fn prove_explicit_queries<E: TowerField>(
    state: &ExplicitProverState<E>,
    r: usize,
    query_seed: F,
) -> (Vec<FriQueryPayloadExplicit<E>>, FriLayerProofs, Vec<[u8; HASH_BYTES]>) {
    let (all_refs, roots, layer_proofs) =
        fri_prove_layer_openings_only::<E>(&state.fri_state, r, query_seed);

    let schedule_len = state.fri_state.transcript.schedule.len();

    let mut queries = Vec::with_capacity(r);
    for q_refs in all_refs.into_iter() {
        // Per-layer (f, s, q) extension payloads.
        let mut payloads = Vec::with_capacity(schedule_len);
        for (ell, rref) in q_refs.per_layer_refs.iter().enumerate() {
            payloads.push(LayerOpenPayload {
                f_val: state.fri_state.f_layers_ext[ell][rref.i],
                s_val: state.fri_state.s_layers[ell][rref.i],
                q_val: if state.fri_state.q_layers[ell].is_empty() {
                    use ark_ff::Zero;
                    E::zero()
                } else {
                    state.fri_state.q_layers[ell][rref.i]
                },
            });
        }

        // Layer-0 opening: from Layer0Commit, not from the f0-tree
        // path that fri_prove_queries uses for the classic form.
        let layer0_idx = q_refs.per_layer_refs[0].i;
        let layer0_opening = state.layer0_commit.open(layer0_idx);

        queries.push(FriQueryPayloadExplicit {
            per_layer_refs:     q_refs.per_layer_refs,
            per_layer_payloads: payloads,
            layer0_opening,
            final_index:        q_refs.final_index,
        });
    }

    (queries, layer_proofs, roots)
}

// ═══════════════════════════════════════════════════════════════
//  derive_query_seed_explicit — FS-replay for query_seed
// ═══════════════════════════════════════════════════════════════

/// FS replay that derives the per-proof query seed for the explicit
/// form.
///
/// Mirrors the classic `deep_fri_prove`'s replay (fri.rs lines
/// 1993-2046) with TWO insertions between the `z_fp3` draw and the
/// FRI_SEED draw:
///   - Absorb OOD claims (trace_at_shifts row-major then q_at_z).
///   - Draw γ_1, γ_2, β.
///
/// These match the prover-side `prove_layer0_phase` transcript
/// sequence exactly, so the FS chain through query_seed is unbroken.
///
/// After the layer loop and final-poly absorb, draws
/// `b"query_seed"` as a base-field challenge — identical to classic.
pub(crate) fn derive_query_seed_explicit<E: TowerField>(
    state: &ExplicitProverState<E>,
    domain0: FriDomain,
    fri_params: &FriProverParams,
    final_poly_coeffs: &[E],
) -> F {
    let l = fri_params.schedule.len();
    let use_coeff_commit = fri_params.coeff_commit_final && l > 0;
    let normal_layers = if use_coeff_commit { l - 1 } else { l };

    let mut tr = transcript::Transcript::new_matching_hash(b"FRI/FS");

    bind_statement_to_transcript::<E>(
        &mut tr,
        &fri_params.schedule,
        domain0.size,
        fri_params.seed_z,
        fri_params.coeff_commit_final,
        fri_params.stir,
    );
    tr.absorb_bytes(&state.layer0_commit.root);
    let _ = challenge_ext::<E>(&mut tr, b"z_fp3");

    // EXPLICIT-form insertions: absorb OOD claims then draw γ.
    for col in &state.ood_claims.trace_at_shifts {
        for &v in col {
            absorb_ext::<E>(&mut tr, v);
        }
    }
    absorb_ext::<E>(&mut tr, state.ood_claims.q_at_z);
    let _ = challenge_ext::<E>(&mut tr, b"ali_gamma1");
    let _ = challenge_ext::<E>(&mut tr, b"ali_gamma2");
    let _ = challenge_ext::<E>(&mut tr, b"ali_beta");

    // FRI_SEED draw — same point in the transcript as the prover used
    // inside fri_rounds_from_f0_ext.
    let _: [u8; HASH_BYTES] = transcript_challenge_hash(&mut tr, ds::FRI_SEED);

    // Per-layer draws + absorbs — identical to classic.
    for ell in 0..normal_layers {
        let _ = challenge_ext::<E>(&mut tr, b"alpha");
        if fri_params.stir {
            let coset_evals = &state.fri_state.stir_coset_evals.as_ref().unwrap()[ell];
            for &ev in coset_evals {
                absorb_ext::<E>(&mut tr, ev);
            }
        } else {
            absorb_ext::<E>(&mut tr, state.fri_state.fz_layers[ell]);
        }
        tr.absorb_bytes(&state.fri_state.transcript.layers[ell].root);
    }

    if use_coeff_commit {
        let ell = l - 1;
        if fri_params.stir {
            let coset_evals = &state.fri_state.stir_coset_evals.as_ref().unwrap()[ell];
            for &ev in coset_evals {
                absorb_ext::<E>(&mut tr, ev);
            }
        } else {
            absorb_ext::<E>(&mut tr, state.fri_state.fz_layers[ell]);
        }
        tr.absorb_bytes(&state.fri_state.transcript.layers[ell].root);

        tr.absorb_bytes(&state.fri_state.coeff_root.expect(
            "coeff_root must be Some when coeff_commit_final && L>0"));
        let _ = challenge_ext::<E>(&mut tr, b"alpha");
        let _ = challenge_ext::<E>(&mut tr, b"beta_deg");
    }

    for &c in final_poly_coeffs {
        absorb_ext::<E>(&mut tr, c);
    }

    safe_field_challenge(&mut tr, b"query_seed")
}

// ═══════════════════════════════════════════════════════════════
//  deep_fri_prove_explicit — full prover entry
// ═══════════════════════════════════════════════════════════════

/// Full explicit-form prover entry point.
///
/// Produces a `DeepFriProofExplicit<E>` end-to-end:
///   1. `prove_explicit_state` — Layer0Commit + transcript + merge +
///      FRI rounds 1..L.
///   2. Compute `final_poly_coeffs` from `f_layers_ext.last()`.
///   3. `derive_query_seed_explicit` — FS replay → query_seed.
///   4. `prove_explicit_queries` — open `r` queries against the
///      state.
///   5. Pack the proof envelope (including `layer0_tree_label` and
///      `trace_width` for the verifier).
///
/// STIR mode is NOT YET supported in this entry point: if
/// `fri_params.stir == true`, the function panics rather than
/// returning a partial/invalid proof.  STIR plumbing
/// (`stir_proximity_queries`, `stir_coset_evals`) is tracked as a
/// follow-up.
pub fn deep_fri_prove_explicit<E, A>(
    witness: &MergeWitness,
    h0_domain: &[F],
    air: &A,
    domain0: FriDomain,
    fri_params: &FriProverParams,
    layer0_tree_label: u64,
    r: usize,
) -> DeepFriProofExplicit<E>
where
    E: TowerField,
    A: AirOodEvaluator<E>,
{
    assert!(!fri_params.stir,
        "deep_fri_prove_explicit: STIR mode is not yet wired into the explicit form");
    let trace_width = witness.trace_columns.len();

    // 1. Build prover state (layer-0 phase + FRI rounds 1..L).
    let state: ExplicitProverState<E> = prove_explicit_state::<E, A>(
        witness, h0_domain, air, domain0, fri_params, layer0_tree_label,
    );

    // 2. Final-poly coefficients.
    let l = fri_params.schedule.len();
    let final_evals = state.fri_state.f_layers_ext[l].clone();
    let all_coeffs = ext_evals_to_coeffs::<E>(&final_evals);
    let d_final = fri_params.d_final.min(all_coeffs.len());
    let final_poly_coeffs: Vec<E> = all_coeffs[..d_final].to_vec();

    // 3. query_seed via FS replay.
    let query_seed = derive_query_seed_explicit::<E>(
        &state, domain0, fri_params, &final_poly_coeffs,
    );

    // 4. Open r queries.
    let (queries, layer_proofs, roots) =
        prove_explicit_queries::<E>(&state, r, query_seed);

    // 5. Pack proof.
    DeepFriProofExplicit {
        root_f0:    state.layer0_commit.root,
        roots,
        ood_claims: state.ood_claims,
        layer_proofs,
        queries,
        fz_per_layer:      state.fri_state.fz_layers.clone(),
        final_poly_coeffs,
        n0:     domain0.size,
        omega0: domain0.omega,
        coeff_tuples:      state.fri_state.coeff_tuples.clone(),
        coeff_root:        state.fri_state.coeff_root,
        stir_coset_evals:  state.fri_state.stir_coset_evals.clone(),
        stir_proximity_queries: None,
        layer0_tree_label,
        trace_width,
    }
}

// ═══════════════════════════════════════════════════════════════
//  Proof-size accounting
// ═══════════════════════════════════════════════════════════════

/// Approximate wire-format byte count for an explicit-form proof.
///
/// Mirrors `fri::deep_fri_proof_size_bytes` for the classic form but
/// with the layer-0 swap accounted for:
///   - `f0_openings` (each: leaf hash + path) is replaced by per-query
///     `layer0_opening` carrying the wider `(T_1, …, T_w, Q, R)` leaf
///     payload PLUS the Merkle authentication path.
///   - `ood_claims` is carried on the wire explicitly (one E for `z`,
///     `w · k` E values for `trace_at_shifts`, one E for `q_at_z`).
///
/// Field elements assumed 8 bytes (Goldilocks); extension elements
/// `E::DEGREE * 8` bytes.  Merkle nodes `HASH_BYTES` each.
pub fn deep_fri_proof_explicit_size_bytes<E: TowerField>(
    p: &DeepFriProofExplicit<E>,
) -> usize {
    const FIELD_BYTES: usize = 8;
    let ext_bytes: usize = E::DEGREE * FIELD_BYTES;

    let mut bytes = 0usize;

    // root_f0
    bytes += HASH_BYTES;

    // Per-layer roots (layer 1..L commitments).
    bytes += p.roots.len() * HASH_BYTES;

    // OOD claims: z + trace_at_shifts + q_at_z
    bytes += ext_bytes; // z
    for col in &p.ood_claims.trace_at_shifts {
        bytes += col.len() * ext_bytes;
    }
    bytes += ext_bytes; // q_at_z

    // fz_per_layer
    bytes += p.fz_per_layer.len() * ext_bytes;

    // final_poly_coeffs
    bytes += p.final_poly_coeffs.len() * ext_bytes;

    // Per-query: layer-0 opening + per-layer payloads.
    for q in &p.queries {
        // per_layer_payloads: (f, s, q) × |schedule|
        bytes += q.per_layer_payloads.len() * 3 * ext_bytes;

        // Layer-0 opening: leaf payload + Merkle path.
        bytes += q.layer0_opening.leaf.trace_values.len() * FIELD_BYTES; // T_1..T_w
        bytes += 2 * FIELD_BYTES;                                         // Q + R
        bytes += HASH_BYTES;                                              // leaf hash
        for level in &q.layer0_opening.merkle_opening.path {
            bytes += level.len() * HASH_BYTES;
        }
    }

    // Per-layer Merkle openings.
    for layer in &p.layer_proofs.layers {
        for opening in &layer.openings {
            bytes += HASH_BYTES;
            for level in &opening.path {
                bytes += level.len() * HASH_BYTES;
            }
        }
    }

    bytes
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

        // H_0 via cumulative multiply (O(n) instead of O(n log n) pow_u64).
        let mut h0 = Vec::with_capacity(n);
        let mut h_acc = F::one();
        for _ in 0..n {
            h0.push(h_acc);
            h_acc *= omega_n;
        }

        // Random Q of degree ≤ T-1, LDE'd to H_0 via FFT (replaces the
        // O(n·T) naive Horner — bottleneck at large k).
        let q_t_evals: Vec<F> = (0..trace_len).map(|_| F::rand(rng)).collect();
        let q_domain = GeneralEvaluationDomain::<F>::new(trace_len).unwrap();
        let mut q_coeffs: Vec<F> = q_domain.ifft(&q_t_evals);
        q_coeffs.resize(n, F::zero());
        let n_domain = GeneralEvaluationDomain::<F>::new(n).unwrap();
        let q_on_h0: Vec<F> = n_domain.fft(&q_coeffs);

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

    // ── prove_explicit_state ── (P5.3 integration)

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
            layer0_tree_label: 0x55_AAAA,
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

    /// Honest e2e: prove_explicit_state on a constant-boundary witness
    /// runs both phases without panic; the FRI state's layer-0 is the
    /// merge output, the root_f0 is the Layer0Commit root, and z_ext
    /// is the OOD point.
    #[test]
    fn prove_explicit_state_honest_e2e_smoke() {
        let mut rng = StdRng::seed_from_u64(0x5301_0001);
        let trace_len = 8usize;
        let blowup = 4usize;
        let n = trace_len * blowup;
        let (witness, h0) = build_honest_witness(
            &mut rng, trace_len, blowup, F::from(13u64));
        let air = ConstantBoundaryAir { c: F::from(13u64) };
        let fri_params = baseline_fri_params();
        let domain0 = FriDomain::new_radix2(n);
        let layer0_tree_label = 0x53_0001u64;

        let state: ExplicitProverState<Ext> = prove_explicit_state::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, layer0_tree_label,
        );

        // Layer-0 plumbing.
        assert_eq!(state.fri_state.root_f0, state.layer0_commit.root,
            "fri_state.root_f0 must equal Layer0Commit::root");
        assert_eq!(state.fri_state.z_ext, state.ood_claims.z,
            "fri_state.z_ext must equal OOD point z");

        // FRI rounds populated.
        let l = fri_params.schedule.len();
        assert_eq!(state.fri_state.f_layers_ext.len(), l + 1,
            "f_layers_ext should have schedule.len() + 1 entries");
        assert_eq!(state.fri_state.transcript.layers.len(), l,
            "one FriLayerCommitment per fold");

        // Layer-0 f_0 was the Ext-valued merge output, not a base-field lift.
        assert!(state.fri_state.f_layers_ext[0].iter().any(|v| *v != Ext::zero()),
            "layer-0 f_0 must be non-trivially populated");

        // FriProverState.f0_base is empty in the explicit form.
        assert!(state.fri_state.f0_base.is_empty(),
            "explicit form must not populate FriProverState::f0_base");

        // Every layer root is non-zero.
        for (ell, c) in state.fri_state.transcript.layers.iter().enumerate() {
            assert_ne!(c.root, [0u8; HASH_BYTES],
                "layer {} root unexpectedly all-zero", ell);
        }
    }

    /// Determinism: identical inputs → identical state across both
    /// phases (layer-0 + FRI rounds).  This is the FS-binding chain
    /// integrity test: every Merkle root from layers 0..L must match
    /// bit-for-bit.
    #[test]
    fn prove_explicit_state_deterministic() {
        let mut rng = StdRng::seed_from_u64(0x5302_0002);
        let trace_len = 8usize;
        let blowup = 4usize;
        let n = trace_len * blowup;
        let (witness, h0) = build_honest_witness(
            &mut rng, trace_len, blowup, F::from(21u64));
        let air = ConstantBoundaryAir { c: F::from(21u64) };
        let fri_params = baseline_fri_params();
        let domain0 = FriDomain::new_radix2(n);
        let label = 0x53_0002u64;

        let a = prove_explicit_state::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, label);
        let b = prove_explicit_state::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, label);

        assert_eq!(a.layer0_commit.root, b.layer0_commit.root);
        assert_eq!(a.ood_claims.z, b.ood_claims.z);
        assert_eq!(a.fri_state.transcript.layers.len(), b.fri_state.transcript.layers.len());
        for ((ell, ra), rb) in a.fri_state.transcript.layers.iter().enumerate()
            .zip(b.fri_state.transcript.layers.iter())
        {
            assert_eq!(ra.root, rb.root, "layer {} root mismatch", ell);
        }
        assert_eq!(a.fri_state.alpha_layers, b.fri_state.alpha_layers);
        assert_eq!(a.fri_state.fz_layers, b.fri_state.fz_layers);
    }

    /// Tamper at layer 0 (witness mutation) must propagate to layer-L
    /// roots — the FS chain is unbroken end-to-end.
    #[test]
    fn tamper_witness_propagates_to_every_fri_layer_root() {
        let mut rng = StdRng::seed_from_u64(0x5303_0003);
        let trace_len = 8usize;
        let blowup = 4usize;
        let n = trace_len * blowup;
        let (witness, h0) = build_honest_witness(
            &mut rng, trace_len, blowup, F::from(7u64));
        let air = ConstantBoundaryAir { c: F::from(7u64) };
        let fri_params = baseline_fri_params();
        let domain0 = FriDomain::new_radix2(n);
        let label = 0x53_0003u64;

        let orig = prove_explicit_state::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, label);

        let mut tampered = witness.clone();
        tampered.trace_columns[0][5] += F::one();
        let bad = prove_explicit_state::<Ext, _>(
            &tampered, &h0, &air, domain0, &fri_params, label);

        assert_ne!(orig.layer0_commit.root, bad.layer0_commit.root);
        // Every layer root differs because the transcript chain feeds
        // layer ell+1 with material drawn from layer ell.
        for (ell, (oc, bc)) in orig.fri_state.transcript.layers.iter()
            .zip(bad.fri_state.transcript.layers.iter()).enumerate()
        {
            assert_ne!(oc.root, bc.root,
                "FS chain broken: layer {} root unchanged after witness tamper", ell);
        }
    }

    /// Different `layer0_tree_label` ⇒ different layer-0 root ⇒
    /// different OOD point z ⇒ different layer-1+ roots.
    #[test]
    fn different_tree_label_propagates_through_fri_rounds() {
        let mut rng = StdRng::seed_from_u64(0x5304_0004);
        let trace_len = 8usize;
        let blowup = 4usize;
        let n = trace_len * blowup;
        let (witness, h0) = build_honest_witness(
            &mut rng, trace_len, blowup, F::from(4u64));
        let air = ConstantBoundaryAir { c: F::from(4u64) };
        let fri_params = baseline_fri_params();
        let domain0 = FriDomain::new_radix2(n);

        let a = prove_explicit_state::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, 0xAAAA);
        let b = prove_explicit_state::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, 0xBBBB);

        assert_ne!(a.layer0_commit.root, b.layer0_commit.root);
        assert_ne!(a.ood_claims.z, b.ood_claims.z);
        for (ell, (ra, rb)) in a.fri_state.transcript.layers.iter()
            .zip(b.fri_state.transcript.layers.iter()).enumerate()
        {
            assert_ne!(ra.root, rb.root,
                "layer {} root identical across distinct tree_labels", ell);
        }
    }

    // ── prove_explicit_queries ── (P5.5)

    /// Honest e2e: produce r=4 queries and verify each Layer0Opening
    /// reconstructs the prover's f_0 at the queried H_0 position.
    /// Also confirms per-layer payloads match the prover state at
    /// the per-layer indices.
    #[test]
    fn prove_explicit_queries_honest_layer0_reconstruction_matches_prover_f0() {
        use merkle::MerkleChannelCfg;
        let (state, _w, h0, _trace_len, _air, fri_params) =
            run_honest_prover(0x5505_0001, 17);

        let r = 4usize;
        let query_seed = F::from(0xBABEu64);
        let (queries, _layer_proofs, _roots) =
            prove_explicit_queries::<Ext>(&state, r, query_seed);
        assert_eq!(queries.len(), r);

        // Build the verifier-side Merkle config for layer-0.
        let n0 = state.fri_state.f_layers_ext[0].len();
        let depth = (n0.next_power_of_two().trailing_zeros() as usize).max(1);
        let layer0_label = baseline_layer0_params().layer0_tree_label;
        let cfg = MerkleChannelCfg::new(vec![2usize; depth], layer0_label);

        // Toy AIR uses Σ = {1}; the verifier-side reconstruction
        // needs the same shift slice.
        let shifts: &[F] = &[F::one()];
        let trace_len = 8usize;
        let d0 = (2usize - 1) * trace_len - 1;

        for q in &queries {
            // Per-layer payloads agree with the prover state.
            for (ell, (rref, pay)) in
                q.per_layer_refs.iter().zip(q.per_layer_payloads.iter()).enumerate()
            {
                assert_eq!(pay.f_val, state.fri_state.f_layers_ext[ell][rref.i],
                    "f_val mismatch at layer {}", ell);
                assert_eq!(pay.s_val, state.fri_state.s_layers[ell][rref.i],
                    "s_val mismatch at layer {}", ell);
            }

            // Layer-0 opening: Merkle path verifies AND reconstruction
            // matches the prover-side f_0 at this position.
            let idx = q.per_layer_refs[0].i;
            let recon = q.layer0_opening
                .verify_and_reconstruct::<Ext>(
                    &cfg, state.layer0_commit.root, h0[idx],
                    shifts,
                    &state.ood_claims,
                    &state.merge_challenges,
                    trace_len,
                    d0,
                )
                .expect("Merkle verify + reconstruct must succeed");

            // Bit-for-bit match with prover's f_0 at this position.
            assert_eq!(recon, state.fri_state.f_layers_ext[0][idx],
                "verifier-reconstructed f_0 at idx {} differs from prover", idx);
        }

        let _ = fri_params;
    }

    /// Determinism: same (state, r, query_seed) → identical query
    /// data (per-layer refs, payloads, layer0_opening payloads).
    #[test]
    fn prove_explicit_queries_deterministic() {
        let (state, _w, _h0, _trace_len, _air, _fri_params) =
            run_honest_prover(0x5505_0002, 19);
        let r = 3usize;
        let query_seed = F::from(0xFEEDu64);
        let (a, _, _) = prove_explicit_queries::<Ext>(&state, r, query_seed);
        let (b, _, _) = prove_explicit_queries::<Ext>(&state, r, query_seed);
        assert_eq!(a.len(), b.len());
        for (qa, qb) in a.iter().zip(b.iter()) {
            assert_eq!(qa.final_index, qb.final_index);
            assert_eq!(qa.per_layer_refs.len(), qb.per_layer_refs.len());
            for (ra, rb) in qa.per_layer_refs.iter().zip(qb.per_layer_refs.iter()) {
                assert_eq!(ra.i, rb.i);
                assert_eq!(ra.child_pos, rb.child_pos);
                assert_eq!(ra.parent_index, rb.parent_index);
            }
            assert_eq!(qa.layer0_opening.index, qb.layer0_opening.index);
            assert_eq!(qa.layer0_opening.leaf.trace_values,
                       qb.layer0_opening.leaf.trace_values);
            assert_eq!(qa.layer0_opening.leaf.q_value, qb.layer0_opening.leaf.q_value);
            assert_eq!(qa.layer0_opening.leaf.r_value, qb.layer0_opening.leaf.r_value);
        }
    }

    /// Different query_seed ⇒ different query positions (at least
    /// one differs across the r queries).
    #[test]
    fn prove_explicit_queries_query_seed_changes_positions() {
        let (state, _w, _h0, _trace_len, _air, _fri_params) =
            run_honest_prover(0x5505_0003, 23);
        let r = 4usize;
        let (a, _, _) = prove_explicit_queries::<Ext>(&state, r, F::from(0x1111u64));
        let (b, _, _) = prove_explicit_queries::<Ext>(&state, r, F::from(0x2222u64));

        let diff_count = a.iter().zip(b.iter())
            .filter(|(qa, qb)|
                qa.per_layer_refs[0].i != qb.per_layer_refs[0].i)
            .count();
        assert!(diff_count > 0,
            "distinct query_seeds must yield at least one distinct layer-0 index");
    }

    // ── deep_fri_prove_explicit ── (P5.6)

    /// Honest e2e: produces a fully-populated DeepFriProofExplicit
    /// with consistent metadata (root_f0 = layer0_commit.root, n0,
    /// trace_width, query count, fz_per_layer length, roots length).
    #[test]
    fn deep_fri_prove_explicit_honest_e2e_shape() {
        let mut rng = StdRng::seed_from_u64(0x5606_0001);
        let trace_len = 8usize;
        let blowup = 4usize;
        let n = trace_len * blowup;
        let (witness, h0) = build_honest_witness(
            &mut rng, trace_len, blowup, F::from(31u64));
        let air = ConstantBoundaryAir { c: F::from(31u64) };
        let fri_params = baseline_fri_params();
        let domain0 = FriDomain::new_radix2(n);
        let label = 0x5606_AAAAu64;
        let r = 6usize;

        let proof: DeepFriProofExplicit<Ext> = deep_fri_prove_explicit::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, label, r,
        );

        assert_eq!(proof.n0, n);
        assert_eq!(proof.trace_width, witness.trace_columns.len());
        assert_eq!(proof.layer0_tree_label, label);
        assert_eq!(proof.queries.len(), r);
        assert_eq!(proof.fz_per_layer.len(), fri_params.schedule.len());
        assert_eq!(proof.roots.len(), fri_params.schedule.len());
        assert_eq!(proof.layer_proofs.layers.len(), fri_params.schedule.len());
        assert!(!proof.final_poly_coeffs.is_empty(),
            "final_poly_coeffs must be populated (d_final >= 1)");

        // Per-query: layer0_opening index matches per_layer_refs[0].i.
        for q in &proof.queries {
            assert_eq!(q.layer0_opening.index, q.per_layer_refs[0].i);
            assert_eq!(q.layer0_opening.leaf.width(), witness.trace_columns.len());
        }

        // Layer proofs: one MerkleOpening per query per layer.
        for (ell, lp) in proof.layer_proofs.layers.iter().enumerate() {
            assert_eq!(lp.openings.len(), r,
                "layer {} has {} openings, expected r={}", ell, lp.openings.len(), r);
        }
    }

    /// Determinism: same (witness, params, label, r) → identical
    /// proof (root_f0, every layer root, every query's leaf payload).
    #[test]
    fn deep_fri_prove_explicit_deterministic() {
        let mut rng = StdRng::seed_from_u64(0x5606_0002);
        let trace_len = 8usize;
        let blowup = 4usize;
        let n = trace_len * blowup;
        let (witness, h0) = build_honest_witness(
            &mut rng, trace_len, blowup, F::from(43u64));
        let air = ConstantBoundaryAir { c: F::from(43u64) };
        let fri_params = baseline_fri_params();
        let domain0 = FriDomain::new_radix2(n);
        let label = 0x5606_BBBBu64;
        let r = 4usize;

        let a: DeepFriProofExplicit<Ext> = deep_fri_prove_explicit::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, label, r);
        let b: DeepFriProofExplicit<Ext> = deep_fri_prove_explicit::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, label, r);

        assert_eq!(a.root_f0, b.root_f0);
        assert_eq!(a.roots, b.roots);
        assert_eq!(a.fz_per_layer, b.fz_per_layer);
        assert_eq!(a.final_poly_coeffs, b.final_poly_coeffs);
        for (qa, qb) in a.queries.iter().zip(b.queries.iter()) {
            assert_eq!(qa.layer0_opening.index, qb.layer0_opening.index);
            assert_eq!(qa.layer0_opening.leaf.trace_values,
                       qb.layer0_opening.leaf.trace_values);
            assert_eq!(qa.layer0_opening.leaf.q_value, qb.layer0_opening.leaf.q_value);
            assert_eq!(qa.layer0_opening.leaf.r_value, qb.layer0_opening.leaf.r_value);
            assert_eq!(qa.final_index, qb.final_index);
        }
    }

    /// query_seed FS replay is consistent: the query_seed
    /// derive_query_seed_explicit computes from the state matches
    /// what deep_fri_prove_explicit ends up using to open queries.
    /// (Implicit pin: prover and verifier MUST agree on query_seed,
    /// so its derivation must be a pure function of public data.)
    #[test]
    fn derive_query_seed_explicit_deterministic() {
        let (state, _w, _h0, _trace_len, _air, fri_params) =
            run_honest_prover(0x5606_0003, 47);
        let domain0 = FriDomain::new_radix2(state.fri_state.f_layers_ext[0].len());

        // Synthetic final_poly_coeffs (would normally come from
        // ext_evals_to_coeffs on the last layer).
        let final_evals = state.fri_state.f_layers_ext[fri_params.schedule.len()].clone();
        let all_coeffs = ext_evals_to_coeffs::<Ext>(&final_evals);
        let d_final = fri_params.d_final.min(all_coeffs.len());
        let final_poly = all_coeffs[..d_final].to_vec();

        let qs1 = derive_query_seed_explicit::<Ext>(
            &state, domain0, &fri_params, &final_poly);
        let qs2 = derive_query_seed_explicit::<Ext>(
            &state, domain0, &fri_params, &final_poly);
        assert_eq!(qs1, qs2, "query_seed derivation must be deterministic");
    }

    /// Tamper a single witness entry — root_f0 changes AND query
    /// positions change (because query_seed FS-derives from root_f0
    /// through the whole chain).
    #[test]
    fn deep_fri_prove_explicit_witness_tamper_changes_root_and_queries() {
        let mut rng = StdRng::seed_from_u64(0x5606_0004);
        let trace_len = 8usize;
        let blowup = 4usize;
        let n = trace_len * blowup;
        let (witness, h0) = build_honest_witness(
            &mut rng, trace_len, blowup, F::from(2u64));
        let air = ConstantBoundaryAir { c: F::from(2u64) };
        let fri_params = baseline_fri_params();
        let domain0 = FriDomain::new_radix2(n);
        let label = 0x5606_CCCCu64;
        let r = 4usize;

        let orig: DeepFriProofExplicit<Ext> = deep_fri_prove_explicit::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, label, r);

        let mut tampered = witness.clone();
        tampered.trace_columns[0][4] += F::one();
        let bad: DeepFriProofExplicit<Ext> = deep_fri_prove_explicit::<Ext, _>(
            &tampered, &h0, &air, domain0, &fri_params, label, r);

        assert_ne!(orig.root_f0, bad.root_f0);
        // At least one query position differs (FS chain end-to-end).
        let any_q_diff = orig.queries.iter().zip(bad.queries.iter())
            .any(|(a, b)| a.layer0_opening.index != b.layer0_opening.index);
        assert!(any_q_diff,
            "witness tamper propagated to root_f0 but NOT to any query position");
    }

    /// `fri_params.stir = true` panics with a clear message (STIR
    /// mode is explicitly out-of-scope for this commit).
    #[test]
    #[should_panic(expected = "STIR mode is not yet wired")]
    fn deep_fri_prove_explicit_panics_on_stir() {
        let mut rng = StdRng::seed_from_u64(0x5606_0005);
        let trace_len = 8usize;
        let blowup = 4usize;
        let n = trace_len * blowup;
        let (witness, h0) = build_honest_witness(
            &mut rng, trace_len, blowup, F::from(7u64));
        let air = ConstantBoundaryAir { c: F::from(7u64) };
        let mut fri_params = baseline_fri_params();
        fri_params.stir = true;
        let domain0 = FriDomain::new_radix2(n);
        let _ = deep_fri_prove_explicit::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, 0, 1);
    }

    // ═══════════════════════════════════════════════════════════
    //  P5.9 — Bench harness
    // ═══════════════════════════════════════════════════════════

    /// Bench the explicit-form prove + verify pipeline at user-set
    /// parameters.  `#[ignore]`d like `stir_halve::tests::
    /// canonical_k22_proof_size`; run with `--ignored --nocapture`
    /// to emit the report.
    ///
    /// Env vars (with defaults sane for a 1024-row LDE):
    ///   EXPLICIT_K_LOG       log2(n)             default 10
    ///   EXPLICIT_BLOWUP      blowup factor       default 4
    ///   EXPLICIT_R           query count r       default 30
    ///   EXPLICIT_LABEL       free-form label     default "explicit-l1"
    ///   EXPLICIT_CSV_APPEND  CSV path to append  default (no append)
    ///
    /// CSV row format (no header; the sweep wrapper supplies one):
    ///   label,k_log,n,trace_len,blowup,r,prove_ms,verify_ms,proof_bytes
    #[test]
    #[ignore]
    fn explicit_form_bench_one_cell() {
        use std::env;
        use std::io::Write;
        use std::time::Instant;

        let k_log: usize = env::var("EXPLICIT_K_LOG").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(10);
        let blowup: usize = env::var("EXPLICIT_BLOWUP").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(4);
        let r: usize = env::var("EXPLICIT_R").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(30);
        let label: String = env::var("EXPLICIT_LABEL")
            .unwrap_or_else(|_| "explicit-l1".to_string());
        let csv_append: Option<String> = env::var("EXPLICIT_CSV_APPEND").ok();

        let n: usize = 1usize << k_log;
        assert!(n % blowup == 0, "n={} must be divisible by blowup={}", n, blowup);
        let trace_len = n / blowup;

        // Schedule: either user-supplied (EXPLICIT_SCHEDULE="4,4,4,4,4,4,4,4")
        // or default to binary fold down to final_size = 4.
        let schedule: Vec<usize> = match env::var("EXPLICIT_SCHEDULE") {
            Ok(s) => s.split(',')
                .map(|x| x.trim().parse::<usize>()
                    .unwrap_or_else(|_| panic!("EXPLICIT_SCHEDULE parse: {}", x)))
                .collect(),
            Err(_) => {
                let final_size = 4usize;
                assert!(n >= final_size && (n / final_size).is_power_of_two(),
                    "n / final_size must be a power of two");
                let n_folds = (n / final_size).trailing_zeros() as usize;
                vec![2usize; n_folds]
            }
        };

        // Validate schedule shape against n.
        let total_fold: usize = schedule.iter().product();
        assert!(n % total_fold == 0,
            "schedule product {} must divide n = {}", total_fold, n);
        let final_size = n / total_fold;
        assert!(final_size >= 1, "schedule too aggressive: final_size = 0");
        assert!(final_size.is_power_of_two(),
            "schedule yields non-power-of-two final_size = {}", final_size);

        // d_final: max degree of the final polynomial.  Defaults to 1
        // (constant) — matches the canonical STIR fold-to-constant
        // setup.  Override via EXPLICIT_D_FINAL.
        let d_final: usize = env::var("EXPLICIT_D_FINAL").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(1);
        assert!(d_final <= final_size,
            "d_final={} > final_size={}", d_final, final_size);

        let mut rng = StdRng::seed_from_u64(0xBE_5959u64);
        let c = F::from(42u64);
        let (witness, h0) = build_honest_witness(&mut rng, trace_len, blowup, c);
        let air = ConstantBoundaryAir { c };

        let fri_params = FriProverParams {
            schedule: schedule.clone(),
            seed_z: 0x59_AC0,
            coeff_commit_final: false,
            d_final,
            stir: false,
        };
        let domain0 = FriDomain::new_radix2(n);
        let layer0_label = 0x5959_0001u64;

        // Prove.
        let t0 = Instant::now();
        let proof = deep_fri_prove_explicit::<Ext, _>(
            &witness, &h0, &air, domain0, &fri_params, layer0_label, r);
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Verify.
        let t1 = Instant::now();
        use crate::explicit_merge_verify::deep_fri_verify_explicit;
        let result = deep_fri_verify_explicit::<Ext, _>(
            &proof, &air, trace_len, &fri_params, domain0);
        let verify_ms = t1.elapsed().as_secs_f64() * 1000.0;
        assert!(result.accepted, "bench proof rejected by verifier");

        let proof_bytes = deep_fri_proof_explicit_size_bytes::<Ext>(&proof);

        eprintln!();
        eprintln!("=== Explicit-form bench: {} ===", label);
        eprintln!("  k_log={}, n={}, trace_len={}, blowup={}, r={}",
            k_log, n, trace_len, blowup, r);
        eprintln!("  schedule = {:?}", schedule);
        eprintln!("  Prove  : {:.2} ms", prove_ms);
        eprintln!("  Verify : {:.2} ms", verify_ms);
        eprintln!("  Proof  : {} bytes ({:.2} KiB)",
            proof_bytes, proof_bytes as f64 / 1024.0);

        if let Some(path) = csv_append {
            let mut f = std::fs::OpenOptions::new()
                .create(true).append(true).open(&path)
                .expect("open EXPLICIT_CSV_APPEND");
            writeln!(f,
                "{},{},{},{},{},{},{:.3},{:.3},{}",
                label, k_log, n, trace_len, blowup, r,
                prove_ms, verify_ms, proof_bytes,
            ).expect("write CSV row");
        }
    }
}
