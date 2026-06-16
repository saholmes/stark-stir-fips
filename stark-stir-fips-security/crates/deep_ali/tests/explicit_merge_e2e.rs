//! P5.8 — End-to-end integration tests through the explicit-form
//! public API.
//!
//! Companion to `tests/explicit_merge_soundness.rs` (P4), which
//! exercised the construction-layer primitives (eq:merge + layer-0
//! commit + eq:ali-check) independently of the FRI machinery.  This
//! file exercises the FULL pipeline through the top-level public
//! entry points only:
//!
//!   - `deep_fri_prove_explicit`   (prover, P5.6)
//!   - `deep_fri_verify_explicit`  (verifier, P5.7)
//!
//! Coverage:
//!   - Section A: honest prover ↔ verifier round-trip.
//!   - Section B: per-tamper-site rejection coverage (the soundness
//!     matrix from P5.7's closeout).
//!   - Section C: parameter / fixture variations (multiple AIR
//!     instances, multiple `r` values, distinct seeds).
//!   - Section D: determinism + replay defence.
//!
//! What this file does NOT verify (deferred):
//!   - FRI fold arithmetic between consecutive layers (the
//!     `(f_pos, f_neg) → expected` chain).
//!   - Final-poly low-degree consistency against the final FRI
//!     layer.
//!
//! Both are layered ON TOP of the prover↔verifier checks already
//! exercised here; their absence does not invalidate the soundness
//! gates this file tests (Merkle binding, reconstruction equality,
//! OOD consistency, FS chain integrity).

use ark_ff::{FftField, Field, One, UniformRand, Zero};
use ark_goldilocks::Goldilocks as F;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use rand::{rngs::StdRng, SeedableRng};

use deep_ali::explicit_merge::MergeWitness;
use deep_ali::explicit_merge_air::AirOodEvaluator;
use deep_ali::explicit_merge_prove::{deep_fri_prove_explicit, DeepFriProofExplicit};
use deep_ali::explicit_merge_verify::deep_fri_verify_explicit;
use deep_ali::fri::{FriDomain, FriProverParams};
use deep_ali::sextic_ext::SexticExt;
use deep_ali::tower_field::TowerField;

type Ext = SexticExt;

// ─── Toy AIR fixture: constant boundary Φ(X) = T(X) - c ──

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

fn baseline_params() -> (FriProverParams, u64) {
    (
        FriProverParams {
            schedule: vec![2, 2, 2],
            seed_z: 0xC0FFEE,
            coeff_commit_final: false,
            d_final: 1,
            stir: false,
            public_inputs_hash: None,
        },
        0x58_AAAA,
    )
}

/// Run an honest prover end-to-end with one (seed, c, r) triple.
fn run_honest(seed: u64, c_val: u64, r: usize)
    -> (DeepFriProofExplicit<Ext>, ConstantBoundaryAir, usize, FriProverParams, FriDomain)
{
    let mut rng = StdRng::seed_from_u64(seed);
    let trace_len = 8usize;
    let blowup = 4usize;
    let n = trace_len * blowup;
    let c = F::from(c_val);
    let (witness, h0) = build_honest_witness(&mut rng, trace_len, blowup, c);
    let air = ConstantBoundaryAir { c };
    let (fri_params, label) = baseline_params();
    let domain0 = FriDomain::new_radix2(n);
    let proof = deep_fri_prove_explicit::<Ext, _>(
        &witness, &h0, &air, domain0, &fri_params, label, r);
    (proof, air, trace_len, fri_params, domain0)
}

// ═══════════════════════════════════════════════════════════════
//  Section A — Honest round-trip
// ═══════════════════════════════════════════════════════════════

/// A.1 The canonical honest round-trip.  Prover output verifies; all
/// gates pass; every per-query result is accepted.
#[test]
fn a1_honest_proof_accepted_through_full_pipeline() {
    let (proof, air, trace_len, fri_params, domain0) =
        run_honest(0x5801_0001, 13, 5);

    let result = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0,
    );
    assert!(result.z_matches);
    assert!(result.ood_consistent);
    assert!(result.query_seed_matches);
    for (qi, q) in result.per_query.iter().enumerate() {
        assert!(q.accepted, "query {} rejected: {:?}", qi, q);
    }
    assert!(result.accepted);
}

/// A.2 Honest acceptance with `r = 1` (minimum non-trivial query
/// count) — boundary-condition pin.
#[test]
fn a2_honest_proof_r_eq_1_accepted() {
    let (proof, air, trace_len, fri_params, domain0) =
        run_honest(0x5802_0002, 17, 1);
    let result = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0,
    );
    assert!(result.accepted);
    assert_eq!(result.per_query.len(), 1);
}

/// A.3 Honest acceptance with multiple AIR instances (different
/// boundary constants).
#[test]
fn a3_honest_proof_accepted_across_distinct_air_instances() {
    for c in &[3u64, 7, 23, 101, 1_000_000_007] {
        let (proof, air, trace_len, fri_params, domain0) =
            run_honest(0x5803_0000 ^ c, *c, 3);
        let result = deep_fri_verify_explicit::<Ext, _>(
            &proof, &air, trace_len, &fri_params, domain0,
        );
        assert!(result.accepted, "c={} not accepted", c);
    }
}

// ═══════════════════════════════════════════════════════════════
//  Section B — Per-tamper-site rejection
// ═══════════════════════════════════════════════════════════════

/// B.1 Tamper `root_f0` ⇒ z_matches false ⇒ rejected.
#[test]
fn b1_tamper_root_f0_rejected() {
    let (mut proof, air, trace_len, fri_params, domain0) =
        run_honest(0xB001_0001, 19, 3);
    proof.root_f0[7] ^= 1;
    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0);
    assert!(!r.z_matches);
    assert!(!r.accepted);
}

/// B.2 Tamper `ood_claims.z` ⇒ z_matches false ⇒ rejected.
#[test]
fn b2_tamper_ood_z_rejected() {
    let (mut proof, air, trace_len, fri_params, domain0) =
        run_honest(0xB002_0002, 23, 3);
    proof.ood_claims.z += Ext::one();
    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0);
    assert!(!r.z_matches);
    assert!(!r.accepted);
}

/// B.3 Tamper `ood_claims.q_at_z` ⇒ ood_consistent false ⇒ rejected.
#[test]
fn b3_tamper_q_at_z_rejected() {
    let (mut proof, air, trace_len, fri_params, domain0) =
        run_honest(0xB003_0003, 29, 3);
    proof.ood_claims.q_at_z += Ext::one();
    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0);
    assert!(r.z_matches);
    assert!(!r.ood_consistent);
    assert!(!r.accepted);
}

/// B.4 Tamper `ood_claims.trace_at_shifts` ⇒ ood_consistent false ⇒
/// rejected.
#[test]
fn b4_tamper_trace_at_shifts_rejected() {
    let (mut proof, air, trace_len, fri_params, domain0) =
        run_honest(0xB004_0004, 31, 3);
    proof.ood_claims.trace_at_shifts[0][0] += Ext::one();
    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0);
    assert!(r.z_matches);
    assert!(!r.ood_consistent);
    assert!(!r.accepted);
}

/// B.5 Tamper a layer-0 leaf component ⇒ layer0_merkle_ok false on
/// the tampered query ⇒ rejected.
#[test]
fn b5_tamper_layer0_leaf_trace_value_rejected() {
    let (mut proof, air, trace_len, fri_params, domain0) =
        run_honest(0xB005_0005, 37, 3);
    proof.queries[1].layer0_opening.leaf.trace_values[0] += F::one();
    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0);
    assert!(!r.per_query[1].layer0_merkle_ok);
    assert!(!r.accepted);
}

/// B.6 Tamper layer-0 leaf Q value ⇒ same gate as B.5.
#[test]
fn b6_tamper_layer0_leaf_q_rejected() {
    let (mut proof, air, trace_len, fri_params, domain0) =
        run_honest(0xB006_0006, 41, 3);
    proof.queries[0].layer0_opening.leaf.q_value += F::one();
    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0);
    assert!(!r.per_query[0].layer0_merkle_ok);
    assert!(!r.accepted);
}

/// B.7 Tamper layer-0 leaf R value ⇒ same gate as B.5.
#[test]
fn b7_tamper_layer0_leaf_r_rejected() {
    let (mut proof, air, trace_len, fri_params, domain0) =
        run_honest(0xB007_0007, 43, 3);
    proof.queries[2].layer0_opening.leaf.r_value += F::one();
    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0);
    assert!(!r.per_query[2].layer0_merkle_ok);
    assert!(!r.accepted);
}

/// B.8 Tamper `per_layer_payloads[0].f_val` ⇒ reconstruction
/// equality fails (the layer-0 leaf is honest, so Merkle still
/// passes; but the proof's claimed f_val no longer matches what the
/// verifier reconstructs).
#[test]
fn b8_tamper_layer0_f_val_payload_rejected() {
    let (mut proof, air, trace_len, fri_params, domain0) =
        run_honest(0xB008_0008, 47, 3);
    proof.queries[0].per_layer_payloads[0].f_val += Ext::one();
    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0);
    assert!(r.per_query[0].layer0_merkle_ok);
    assert!(!r.per_query[0].layer0_recon_matches_f_val);
    assert!(!r.accepted);
}

/// B.9 Tamper `per_layer_payloads[1].f_val` ⇒ per-layer Merkle
/// recompute mismatches the path's leaf ⇒ rejected at layer 1.
#[test]
fn b9_tamper_layer1_f_val_payload_rejected() {
    let (mut proof, air, trace_len, fri_params, domain0) =
        run_honest(0xB009_0009, 53, 3);
    proof.queries[1].per_layer_payloads[1].f_val += Ext::one();
    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0);
    assert!(!r.per_query[1].per_layer_merkle_ok[1]);
    assert!(!r.accepted);
}

/// B.10 Tamper a per-layer Merkle opening path byte ⇒ per-layer
/// Merkle verifies false ⇒ rejected.
#[test]
fn b10_tamper_per_layer_merkle_path_rejected() {
    let (mut proof, air, trace_len, fri_params, domain0) =
        run_honest(0xB010_0010, 59, 3);
    // Flip one byte of the layer-0 Merkle path for query 0.
    if !proof.layer_proofs.layers[0].openings[0].path.is_empty()
        && !proof.layer_proofs.layers[0].openings[0].path[0].is_empty()
    {
        proof.layer_proofs.layers[0].openings[0].path[0][0][0] ^= 1;
    }
    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0);
    assert!(!r.per_query[0].per_layer_merkle_ok[0]);
    assert!(!r.accepted);
}

// ═══════════════════════════════════════════════════════════════
//  Section C — Parameter variations
// ═══════════════════════════════════════════════════════════════

/// C.1 Verifier with a SCHEDULE different from the prover's ⇒
/// either a soft reject (`!accepted`) OR a hard panic from a
/// shape-mismatched downstream Merkle verify.  Both are acceptable
/// — what matters is the proof never accepts.
#[test]
fn c1_verifier_schedule_mismatch_rejected() {
    use std::panic::AssertUnwindSafe;

    let (proof, air, trace_len, _good, domain0) =
        run_honest(0xC001_0001, 61, 2);
    let bad_params = FriProverParams {
        schedule: vec![4, 2, 2],
        seed_z: 0xC0FFEE,
        coeff_commit_final: false,
        d_final: 1,
        stir: false,
        public_inputs_hash: None,
    };

    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        deep_fri_verify_explicit::<Ext, _>(
            &proof, &air, trace_len, &bad_params, domain0)
    }));

    match outcome {
        Err(_) => { /* shape-panic; acceptable. */ }
        Ok(r) => {
            // Soft-reject path.
            assert!(!r.accepted,
                "schedule mismatch produced an accepted proof");
        }
    }
}

/// C.2 Verifier with WRONG seed_z ⇒ z_matches false.
#[test]
fn c2_verifier_seed_z_mismatch_rejected() {
    let (proof, air, trace_len, _good, domain0) =
        run_honest(0xC002_0002, 67, 2);
    let bad_params = FriProverParams {
        seed_z: 0xDEADBEEF,
        ..baseline_params().0
    };
    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &bad_params, domain0);
    assert!(!r.z_matches);
    assert!(!r.accepted);
}

/// C.3 Verifier with WRONG trace_len ⇒ Z_H(z) differs ⇒
/// ood_consistent false.
#[test]
fn c3_verifier_wrong_trace_len_rejected() {
    let (proof, air, _trace_len, fri_params, domain0) =
        run_honest(0xC003_0003, 71, 2);
    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, /* wrong */ 16, &fri_params, domain0);
    assert!(r.z_matches);
    assert!(!r.ood_consistent);
    assert!(!r.accepted);
}

// ═══════════════════════════════════════════════════════════════
//  Section D — Determinism + replay defence
// ═══════════════════════════════════════════════════════════════

/// D.1 Determinism: same proof verified twice → identical result
/// (proves verifier is a pure function).
#[test]
fn d1_verify_deterministic() {
    let (proof, air, trace_len, fri_params, domain0) =
        run_honest(0xD001_0001, 73, 3);

    let a = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0);
    let b = deep_fri_verify_explicit::<Ext, _>(
        &proof, &air, trace_len, &fri_params, domain0);
    assert_eq!(a.accepted, b.accepted);
    assert_eq!(a.z_matches, b.z_matches);
    assert_eq!(a.ood_consistent, b.ood_consistent);
    assert_eq!(a.query_seed_matches, b.query_seed_matches);
    for (qa, qb) in a.per_query.iter().zip(b.per_query.iter()) {
        assert_eq!(qa.accepted, qb.accepted);
        assert_eq!(qa.layer0_merkle_ok, qb.layer0_merkle_ok);
        assert_eq!(qa.layer0_recon_matches_f_val,
                   qb.layer0_recon_matches_f_val);
    }
}

/// D.2 Prover determinism: same (witness, params, label, r) → same
/// proof (root_f0 + per-query openings).
#[test]
fn d2_prove_deterministic() {
    let (a, _, _, _, _) = run_honest(0xD002_0002, 79, 4);
    let (b, _, _, _, _) = run_honest(0xD002_0002, 79, 4);
    assert_eq!(a.root_f0, b.root_f0);
    assert_eq!(a.roots, b.roots);
    assert_eq!(a.final_poly_coeffs, b.final_poly_coeffs);
    for (qa, qb) in a.queries.iter().zip(b.queries.iter()) {
        assert_eq!(qa.layer0_opening.index, qb.layer0_opening.index);
        assert_eq!(qa.layer0_opening.leaf.trace_values,
                   qb.layer0_opening.leaf.trace_values);
        assert_eq!(qa.layer0_opening.leaf.q_value, qb.layer0_opening.leaf.q_value);
        assert_eq!(qa.layer0_opening.leaf.r_value, qb.layer0_opening.leaf.r_value);
    }
}

/// D.3 Cross-proof replay rejection: opening one proof's
/// Layer0Opening into another proof's verifier context fails because
/// the leaf hash recomputes against a different root_f0.
#[test]
fn d3_cross_proof_replay_rejected() {
    let (mut proof_a, air, trace_len, fri_params, domain0) =
        run_honest(0xD003_A001, 83, 3);
    let (proof_b, _, _, _, _) = run_honest(0xD003_B001, 89, 3);

    // Swap a layer-0 opening between proofs.
    proof_a.queries[0].layer0_opening = proof_b.queries[0].layer0_opening.clone();

    let r = deep_fri_verify_explicit::<Ext, _>(
        &proof_a, &air, trace_len, &fri_params, domain0);
    // Either Merkle path fails OR reconstruction-equality fails;
    // either way the query and the proof are rejected.
    assert!(!r.per_query[0].accepted);
    assert!(!r.accepted);
}

// ═══════════════════════════════════════════════════════════════
//  Section E — Coverage matrix summary
// ═══════════════════════════════════════════════════════════════

/// E.1 Soundness coverage matrix: every documented tamper site is
/// caught by SOMETHING, and the top-level `accepted` flag rejects.
///
/// This is the single-point summary of Section B that an auditor can
/// read at a glance.  If any site goes from "rejects" to "accepts"
/// after a code change, this test fires immediately.
#[test]
fn e1_soundness_coverage_matrix() {
    let make = || run_honest(0xE001_0001, 97, 3);
    let (_, air_ref, trace_len, fri_params, domain0) = make();

    let cases: Vec<(&str, Box<dyn Fn() -> DeepFriProofExplicit<Ext>>)> = vec![
        ("root_f0",          Box::new(|| { let (mut p,_,_,_,_) = make(); p.root_f0[0] ^= 1; p })),
        ("ood_claims.z",     Box::new(|| { let (mut p,_,_,_,_) = make(); p.ood_claims.z += Ext::one(); p })),
        ("ood_claims.q_at_z", Box::new(|| { let (mut p,_,_,_,_) = make(); p.ood_claims.q_at_z += Ext::one(); p })),
        ("ood_claims.trace_at_shifts",
                              Box::new(|| { let (mut p,_,_,_,_) = make(); p.ood_claims.trace_at_shifts[0][0] += Ext::one(); p })),
        ("layer0 leaf T",    Box::new(|| { let (mut p,_,_,_,_) = make(); p.queries[0].layer0_opening.leaf.trace_values[0] += F::one(); p })),
        ("layer0 leaf Q",    Box::new(|| { let (mut p,_,_,_,_) = make(); p.queries[0].layer0_opening.leaf.q_value += F::one(); p })),
        ("layer0 leaf R",    Box::new(|| { let (mut p,_,_,_,_) = make(); p.queries[0].layer0_opening.leaf.r_value += F::one(); p })),
        ("layer-0 f_val",    Box::new(|| { let (mut p,_,_,_,_) = make(); p.queries[0].per_layer_payloads[0].f_val += Ext::one(); p })),
        ("layer-1 f_val",    Box::new(|| { let (mut p,_,_,_,_) = make(); p.queries[0].per_layer_payloads[1].f_val += Ext::one(); p })),
    ];

    for (label, mut_proof) in cases {
        let p = mut_proof();
        let r = deep_fri_verify_explicit::<Ext, _>(
            &p, &air_ref, trace_len, &fri_params, domain0);
        assert!(!r.accepted, "tamper at {} was NOT rejected", label);
    }
}
