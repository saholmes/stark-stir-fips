//! P4 — Soundness-test plumbing.
//!
//! End-to-end tamper tests that exercise the three explicit-merge
//! primitives together:
//!
//!   1. `deep_ali_merge_explicit`            (P1: eq:merge prover)
//!   2. `Layer0Commit` / `Layer0Opening`     (P2: layer-0 binding)
//!   3. `check_ood_consistency`              (P3: eq:ali-check)
//!
//! ## Sections
//!
//! - **A.** Construction-level binding: round-trip equality between
//!   the prover's `MergeOutput::f0_evals_ext` and the verifier-side
//!   `reconstruct_f0_at` over the entire `H_0` domain.  Tampers on
//!   witness fields (T, Q, R) change the Merkle root; tampers on
//!   FS challenges (γ_1, γ_2, β) or OOD claims (trace_at_shifts, z,
//!   q_at_z) change the verifier-side reconstruction.
//!
//! - **B.** OOD consistency: synthetic `q_at_z` derived from a
//!   chosen `constraint_at_z = q_at_z · Z_H(z)`; the check accepts
//!   the honest pair and rejects pairwise tampers.
//!
//! - **C.** Cross-layer composition: a tampered `q_at_z` survives the
//!   OOD consistency check ONLY if `constraint_at_z` is tampered the
//!   matching way (and vice versa) — confirming the two equations
//!   bind a single algebraic relation, not two independent ones.
//!
//! Uses only the public APIs of `deep_ali::explicit_merge`,
//! `deep_ali::explicit_merge_layer0`, and
//! `deep_ali::explicit_merge_ood_check`.

use ark_ff::{FftField, One, UniformRand};
use ark_goldilocks::Goldilocks as F;
use hash::HASH_BYTES;
use merkle::MerkleChannelCfg;
use rand::{rngs::StdRng, Rng, SeedableRng};

use deep_ali::explicit_merge::{
    deep_ali_merge_explicit, reconstruct_f0_at, MergeChallenges, MergeOutput,
    MergeWitness, OodClaims,
};
use deep_ali::explicit_merge_layer0::Layer0Commit;
use deep_ali::explicit_merge_ood_check::{
    check_ood_consistency, check_ood_consistency_from_claims, vanishing_at_ext,
};
use deep_ali::sextic_ext::SexticExt;
use deep_ali::tower_field::TowerField;

type Ext = SexticExt;

// ─── Shared fixture: a synthetic witness + OOD claims ────────────

/// Bundle of inputs the prover sees plus the FS-derived challenges
/// and OOD claims.  Used as the "honest baseline" by every tamper
/// test; each test mutates one field of this bundle.
#[derive(Clone)]
struct Bundle {
    witness:    MergeWitness,
    h0_domain:  Vec<F>,
    shifts:     Vec<F>,
    ood:        OodClaims<Ext>,
    ch:         MergeChallenges<Ext>,
    tree_label: u64,
}

/// Build a synthetic, random bundle suitable for round-trip tests.
/// All field values are random; we test the algebraic *consistency*
/// of the merge construction, not AIR semantics.
fn build_bundle(seed: u64) -> Bundle {
    let mut rng = StdRng::seed_from_u64(seed);
    let trace_len: usize = 8;
    let blowup:    usize = 4;
    let n:         usize = trace_len * blowup;
    let w:         usize = 3;
    let k_shifts:  usize = 2;
    let d_c:       usize = 2;
    let d0:        usize = (d_c - 1) * trace_len - 1;

    let omega: F = F::get_root_of_unity(n as u64).expect("two-adic root");
    let h0: Vec<F> = (0..n).map(|i| omega.pow_u64(i as u64)).collect();

    let witness = MergeWitness {
        trace_columns: (0..w).map(|_| (0..n).map(|_| F::rand(&mut rng)).collect()).collect(),
        ali_quotient:  (0..n).map(|_| F::rand(&mut rng)).collect(),
        blinder:       (0..n).map(|_| F::rand(&mut rng)).collect(),
        trace_len, d0, k_shifts,
    };

    let shifts: Vec<F> = vec![F::one(), omega.pow_u64(blowup as u64)]; // {1, ω_T}

    let z = ext_random(&mut rng);
    let ood = OodClaims {
        z,
        trace_at_shifts: (0..w)
            .map(|_| (0..k_shifts).map(|_| ext_random(&mut rng)).collect())
            .collect(),
        q_at_z: ext_random(&mut rng),
    };
    let ch = MergeChallenges {
        gamma_1: ext_random(&mut rng),
        gamma_2: ext_random(&mut rng),
        beta:    ext_random(&mut rng),
    };

    Bundle { witness, h0_domain: h0, shifts, ood, ch, tree_label: 0x5040_0001 }
}

fn ext_random<R: Rng>(rng: &mut R) -> Ext {
    Ext::from_fp_components(&[
        F::rand(rng), F::rand(rng), F::rand(rng),
        F::rand(rng), F::rand(rng), F::rand(rng),
    ]).expect("ext from components")
}

/// Run the honest prover pipeline end-to-end: merge → layer-0 commit.
fn run_prover(b: &Bundle) -> (MergeOutput<Ext>, Layer0Commit) {
    let out = deep_ali_merge_explicit::<Ext>(
        &b.witness, &b.h0_domain, &b.shifts, &b.ood, &b.ch,
    );
    let commit = Layer0Commit::from_witness(&b.witness, b.tree_label);
    (out, commit)
}

/// Build the Merkle config matching `Layer0Commit::from_witness`.
fn merkle_cfg(n: usize, tree_label: u64) -> MerkleChannelCfg {
    let depth = (n.next_power_of_two().trailing_zeros() as usize).max(1);
    MerkleChannelCfg::new(vec![2usize; depth], tree_label)
}

// ═══════════════════════════════════════════════════════════════
//  Section A — Construction-level binding
// ═══════════════════════════════════════════════════════════════

/// A.1 Full positional equivalence: the prover-side `f_0_evals_ext`
/// matches the verifier-side `reconstruct_f0_at` from each opened
/// layer-0 leaf at every position in `H_0`.
///
/// This is the load-bearing soundness pin: the layer-0 commit binds
/// the prover to a unique f_0 on H_0, and the verifier can
/// independently recompute that f_0 from the opened components
/// (T_1, …, T_w, Q, R) plus the public OOD claims and FS challenges.
#[test]
fn a1_layer0_open_matches_prover_f0_at_every_position() {
    let b = build_bundle(0xA001_0001);
    let (out, commit) = run_prover(&b);
    let cfg = merkle_cfg(b.h0_domain.len(), b.tree_label);

    for i in 0..b.h0_domain.len() {
        let opening = commit.open(i);

        let recon = opening.verify_and_reconstruct::<Ext>(
            &cfg, commit.root, b.h0_domain[i],
            &b.shifts, &b.ood, &b.ch, b.witness.trace_len, b.witness.d0,
        ).expect("Merkle verify + reconstruction succeeds");

        assert_eq!(recon, out.f0_evals_ext[i],
            "f_0 mismatch at i={} (h_0[i]={:?})", i, b.h0_domain[i]);
    }
}

/// A.2 Tamper a single trace column entry — Merkle root changes.
#[test]
fn a2_tamper_t_column_changes_merkle_root() {
    let b = build_bundle(0xA002_0002);
    let (_out_orig, commit_orig) = run_prover(&b);

    let mut b2 = b.clone();
    b2.witness.trace_columns[1][5] += F::one();
    let commit_tampered = Layer0Commit::from_witness(&b2.witness, b2.tree_label);

    assert_ne!(commit_orig.root, commit_tampered.root,
        "tampering T_1[5] must change the layer-0 Merkle root");
}

/// A.3 Tamper a single Q entry — Merkle root changes.
#[test]
fn a3_tamper_q_witness_changes_merkle_root() {
    let b = build_bundle(0xA003_0003);
    let (_out_orig, commit_orig) = run_prover(&b);

    let mut b2 = b.clone();
    b2.witness.ali_quotient[11] += F::one();
    let commit_tampered = Layer0Commit::from_witness(&b2.witness, b2.tree_label);

    assert_ne!(commit_orig.root, commit_tampered.root,
        "tampering Q[11] must change the layer-0 Merkle root");
}

/// A.4 Tamper a single R entry — Merkle root changes.
#[test]
fn a4_tamper_r_witness_changes_merkle_root() {
    let b = build_bundle(0xA004_0004);
    let (_out_orig, commit_orig) = run_prover(&b);

    let mut b2 = b.clone();
    b2.witness.blinder[3] += F::one();
    let commit_tampered = Layer0Commit::from_witness(&b2.witness, b2.tree_label);

    assert_ne!(commit_orig.root, commit_tampered.root,
        "tampering R[3] must change the layer-0 Merkle root");
}

/// A.5 Tamper γ_1 (trace summand scalar) — verifier-side
/// reconstruction differs from the prover-side f_0 at some position.
#[test]
fn a5_tamper_gamma1_changes_reconstruction() {
    let b = build_bundle(0xA005_0005);
    let (out, commit) = run_prover(&b);
    let cfg = merkle_cfg(b.h0_domain.len(), b.tree_label);

    let mut ch_tampered = b.ch.clone();
    ch_tampered.gamma_1 += Ext::one();

    let mut divergences = 0usize;
    for i in 0..b.h0_domain.len() {
        let opening = commit.open(i);
        // Merkle path still verifies (witness unchanged).
        assert!(opening.verify_merkle(&cfg, commit.root));
        let recon_tampered = reconstruct_f0_at::<Ext>(
            &opening.leaf, b.h0_domain[i],
            &b.shifts, &b.ood, &ch_tampered,
            b.witness.trace_len, b.witness.d0,
        );
        if recon_tampered != out.f0_evals_ext[i] { divergences += 1; }
    }
    assert!(divergences > 0,
        "tampering γ_1 must change reconstruction at ≥1 position");
}

/// A.6 Tamper γ_2 (Q summand scalar) — same divergence pattern.
#[test]
fn a6_tamper_gamma2_changes_reconstruction() {
    let b = build_bundle(0xA006_0006);
    let (out, commit) = run_prover(&b);

    let mut ch_tampered = b.ch.clone();
    ch_tampered.gamma_2 += Ext::one();

    let mut divergences = 0usize;
    for i in 0..b.h0_domain.len() {
        let opening = commit.open(i);
        let recon_tampered = reconstruct_f0_at::<Ext>(
            &opening.leaf, b.h0_domain[i],
            &b.shifts, &b.ood, &ch_tampered,
            b.witness.trace_len, b.witness.d0,
        );
        if recon_tampered != out.f0_evals_ext[i] { divergences += 1; }
    }
    assert!(divergences > 0, "tampering γ_2 must change reconstruction");
}

/// A.7 Tamper β (blinder scalar) — same divergence pattern.
#[test]
fn a7_tamper_beta_changes_reconstruction() {
    let b = build_bundle(0xA007_0007);
    let (out, commit) = run_prover(&b);

    let mut ch_tampered = b.ch.clone();
    ch_tampered.beta += Ext::one();

    let mut divergences = 0usize;
    for i in 0..b.h0_domain.len() {
        let opening = commit.open(i);
        let recon_tampered = reconstruct_f0_at::<Ext>(
            &opening.leaf, b.h0_domain[i],
            &b.shifts, &b.ood, &ch_tampered,
            b.witness.trace_len, b.witness.d0,
        );
        if recon_tampered != out.f0_evals_ext[i] { divergences += 1; }
    }
    assert!(divergences > 0, "tampering β must change reconstruction");
}

/// A.8 Tamper a single OOD trace claim T̂_col(σ_i z) — Ĩ_col(x)
/// changes at every x ∈ H_0, so reconstructed f_0 differs.
#[test]
fn a8_tamper_trace_at_shifts_changes_reconstruction() {
    let b = build_bundle(0xA008_0008);
    let (out, commit) = run_prover(&b);

    let mut ood_tampered = b.ood.clone();
    ood_tampered.trace_at_shifts[0][1] += Ext::one();

    let mut divergences = 0usize;
    for i in 0..b.h0_domain.len() {
        let opening = commit.open(i);
        let recon_tampered = reconstruct_f0_at::<Ext>(
            &opening.leaf, b.h0_domain[i],
            &b.shifts, &ood_tampered, &b.ch,
            b.witness.trace_len, b.witness.d0,
        );
        if recon_tampered != out.f0_evals_ext[i] { divergences += 1; }
    }
    assert!(divergences > 0,
        "tampering trace_at_shifts must change reconstruction (via Ĩ)");
}

/// A.9 Tamper OOD `z` itself — V_{zΣ}(x) and Ĩ both shift, so
/// reconstruction differs.
#[test]
fn a9_tamper_ood_z_changes_reconstruction() {
    let b = build_bundle(0xA009_0009);
    let (out, commit) = run_prover(&b);

    let mut ood_tampered = b.ood.clone();
    ood_tampered.z += Ext::one();

    let mut divergences = 0usize;
    for i in 0..b.h0_domain.len() {
        let opening = commit.open(i);
        let recon_tampered = reconstruct_f0_at::<Ext>(
            &opening.leaf, b.h0_domain[i],
            &b.shifts, &ood_tampered, &b.ch,
            b.witness.trace_len, b.witness.d0,
        );
        if recon_tampered != out.f0_evals_ext[i] { divergences += 1; }
    }
    assert!(divergences > 0, "tampering z must change reconstruction");
}

/// A.10 Tamper OOD `q_at_z` — reconstruction differs (Q summand
/// pulls in `q_at_z` directly).
#[test]
fn a10_tamper_q_at_z_changes_reconstruction() {
    let b = build_bundle(0xA00A_000A);
    let (out, commit) = run_prover(&b);

    let mut ood_tampered = b.ood.clone();
    ood_tampered.q_at_z += Ext::one();

    let mut divergences = 0usize;
    for i in 0..b.h0_domain.len() {
        let opening = commit.open(i);
        let recon_tampered = reconstruct_f0_at::<Ext>(
            &opening.leaf, b.h0_domain[i],
            &b.shifts, &ood_tampered, &b.ch,
            b.witness.trace_len, b.witness.d0,
        );
        if recon_tampered != out.f0_evals_ext[i] { divergences += 1; }
    }
    assert!(divergences > 0,
        "tampering q_at_z must change reconstruction (Q summand)");
}

// ═══════════════════════════════════════════════════════════════
//  Section B — OOD consistency (eq:ali-check)
// ═══════════════════════════════════════════════════════════════

/// B.1 Honest synthesised `(q_at_z, constraint_at_z)` accepted.
///
/// Builds a bundle, then OVERRIDES `q_at_z` and synthesises
/// `constraint_at_z = q_at_z · Z_H(z)` so the algebraic equation
/// holds.  The check accepts.
#[test]
fn b1_consistent_synthetic_pair_accepted() {
    let mut b = build_bundle(0xB001_0001);
    let z_h = vanishing_at_ext::<Ext>(b.ood.z, b.witness.trace_len);
    let mut rng = StdRng::seed_from_u64(0xB001_DEAD);
    let q = ext_random(&mut rng);
    let phi = q * z_h;
    b.ood.q_at_z = q;
    assert!(check_ood_consistency_from_claims::<Ext>(
        &b.ood, phi, b.witness.trace_len,
    ));
}

/// B.2 Tampered `q_at_z` rejected (with `constraint_at_z` unchanged).
#[test]
fn b2_tampered_q_at_z_rejected() {
    let mut b = build_bundle(0xB002_0002);
    let z_h = vanishing_at_ext::<Ext>(b.ood.z, b.witness.trace_len);
    let mut rng = StdRng::seed_from_u64(0xB002_BEEF);
    let q = ext_random(&mut rng);
    let phi = q * z_h;
    b.ood.q_at_z = q;
    assert!(check_ood_consistency_from_claims::<Ext>(
        &b.ood, phi, b.witness.trace_len,
    ));

    // Now tamper q_at_z.
    b.ood.q_at_z += Ext::one();
    assert!(!check_ood_consistency_from_claims::<Ext>(
        &b.ood, phi, b.witness.trace_len,
    ));
}

/// B.3 Tampered `constraint_at_z` rejected (`q_at_z` unchanged).
#[test]
fn b3_tampered_constraint_at_z_rejected() {
    let mut b = build_bundle(0xB003_0003);
    let z_h = vanishing_at_ext::<Ext>(b.ood.z, b.witness.trace_len);
    let mut rng = StdRng::seed_from_u64(0xB003_CAFE);
    let q = ext_random(&mut rng);
    let phi = q * z_h;
    b.ood.q_at_z = q;

    assert!(check_ood_consistency_from_claims::<Ext>(
        &b.ood, phi, b.witness.trace_len,
    ));

    // Tamper Φ — simulate a buggy AIR evaluator.
    let bad_phi = phi + Ext::one();
    assert!(!check_ood_consistency_from_claims::<Ext>(
        &b.ood, bad_phi, b.witness.trace_len,
    ));
}

/// B.4 Replaying a valid `(q, Φ)` pair from one bundle against
/// another bundle's `z` is rejected, because `Z_H(z')` differs.
#[test]
fn b4_replay_pair_against_different_z_rejected() {
    let mut rng = StdRng::seed_from_u64(0xB004_F00D);
    let trace_len: usize = 16;

    let z1 = ext_random(&mut rng);
    let z2 = ext_random(&mut rng);
    assert_ne!(z1, z2);

    let q = ext_random(&mut rng);
    let z_h_1 = vanishing_at_ext::<Ext>(z1, trace_len);
    let phi_1 = q * z_h_1;

    // Valid against z1.
    assert!(check_ood_consistency::<Ext>(z1, q, phi_1, trace_len));
    // Invalid replay against z2.
    assert!(!check_ood_consistency::<Ext>(z2, q, phi_1, trace_len));
}

// ═══════════════════════════════════════════════════════════════
//  Section C — Cross-layer composition
// ═══════════════════════════════════════════════════════════════

/// C.1 A tampered `q_at_z` survives the OOD consistency check ONLY
/// if `constraint_at_z` is tampered the matching way.  This pins the
/// fact that the two equations bind a SINGLE algebraic relation, so a
/// dishonest prover cannot independently nudge one side.
#[test]
fn c1_tamper_q_requires_matching_phi_tamper_to_pass_ood_check() {
    let mut rng = StdRng::seed_from_u64(0xC001_BABE);
    let trace_len: usize = 8;
    let z = ext_random(&mut rng);
    let q = ext_random(&mut rng);
    let z_h = vanishing_at_ext::<Ext>(z, trace_len);
    let phi = q * z_h;

    // Honest accept.
    assert!(check_ood_consistency::<Ext>(z, q, phi, trace_len));

    // Tamper q by δ.
    let delta = ext_random(&mut rng);
    let q_bad = q + delta;
    let phi_unchanged = phi;
    assert!(!check_ood_consistency::<Ext>(z, q_bad, phi_unchanged, trace_len));

    // The only Φ that makes (q_bad, Φ) pass is the matching shift
    // δ · Z_H(z).  Anything else rejected.
    let phi_matched = q_bad * z_h;       // = phi + delta · z_h
    assert!(check_ood_consistency::<Ext>(z, q_bad, phi_matched, trace_len));

    // A near-miss Φ that's off by even one extension element rejects.
    let phi_near_miss = phi_matched + Ext::one();
    assert!(!check_ood_consistency::<Ext>(z, q_bad, phi_near_miss, trace_len));
}

/// C.2 Cross-layer: a tamper of any single primitive-input field
/// — witness T, witness Q, witness R, γ_1, γ_2, β, q_at_z, z,
/// trace_at_shifts — affects EITHER the Merkle root OR the
/// verifier-side reconstruction, NEVER neither.  We enumerate all
/// nine tamper sites and assert per-site visibility.
#[test]
fn c2_every_tamper_site_visible_at_some_layer() {
    let b = build_bundle(0xC002_F00D);
    let (out, commit) = run_prover(&b);
    let cfg = merkle_cfg(b.h0_domain.len(), b.tree_label);

    // Witness-layer tampers — root changes (caught by Merkle path).
    for desc in &[
        ("T_2[0]", "trace_columns[2][0]"),
        ("Q[6]",   "ali_quotient[6]"),
        ("R[7]",   "blinder[7]"),
    ] {
        let mut b2 = b.clone();
        match desc.1 {
            "trace_columns[2][0]" => b2.witness.trace_columns[2][0] += F::one(),
            "ali_quotient[6]"     => b2.witness.ali_quotient[6]    += F::one(),
            "blinder[7]"          => b2.witness.blinder[7]         += F::one(),
            _ => unreachable!(),
        }
        let commit2 = Layer0Commit::from_witness(&b2.witness, b2.tree_label);
        assert_ne!(commit.root, commit2.root,
            "witness tamper at {} did not change the Merkle root", desc.0);
    }

    // FS-challenge tampers — reconstruction differs at some position.
    for (label, mutate) in &[
        ("γ_1", 0u8),
        ("γ_2", 1u8),
        ("β",   2u8),
    ] {
        let mut ch2 = b.ch.clone();
        match mutate {
            0 => ch2.gamma_1 += Ext::one(),
            1 => ch2.gamma_2 += Ext::one(),
            2 => ch2.beta    += Ext::one(),
            _ => unreachable!(),
        }
        let mut diverges = false;
        for i in 0..b.h0_domain.len() {
            let opening = commit.open(i);
            assert!(opening.verify_merkle(&cfg, commit.root));
            let recon = reconstruct_f0_at::<Ext>(
                &opening.leaf, b.h0_domain[i],
                &b.shifts, &b.ood, &ch2,
                b.witness.trace_len, b.witness.d0,
            );
            if recon != out.f0_evals_ext[i] { diverges = true; break; }
        }
        assert!(diverges,
            "FS-challenge tamper at {} did not diverge reconstruction", label);
    }

    // OOD-claim tampers — reconstruction differs.
    for (label, mutate) in &[
        ("q_at_z",            0u8),
        ("z",                 1u8),
        ("trace_at_shifts",   2u8),
    ] {
        let mut ood2 = b.ood.clone();
        match mutate {
            0 => ood2.q_at_z += Ext::one(),
            1 => ood2.z      += Ext::one(),
            2 => ood2.trace_at_shifts[0][0] += Ext::one(),
            _ => unreachable!(),
        }
        let mut diverges = false;
        for i in 0..b.h0_domain.len() {
            let opening = commit.open(i);
            let recon = reconstruct_f0_at::<Ext>(
                &opening.leaf, b.h0_domain[i],
                &b.shifts, &ood2, &b.ch,
                b.witness.trace_len, b.witness.d0,
            );
            if recon != out.f0_evals_ext[i] { diverges = true; break; }
        }
        assert!(diverges,
            "OOD tamper at {} did not diverge reconstruction", label);
    }
}

/// C.3 Construction is deterministic in its public inputs: rebuilding
/// from the same `Bundle` yields the same Merkle root and the same
/// `f_0_evals_ext`.  Detects a stale-state regression (e.g. a hidden
/// RNG inside the prover).
#[test]
fn c3_construction_deterministic_in_public_inputs() {
    let b = build_bundle(0xC003_DEAD);
    let (out_a, commit_a) = run_prover(&b);
    let (out_b, commit_b) = run_prover(&b);

    assert_eq!(commit_a.root, commit_b.root,
        "Layer0Commit root not deterministic in (witness, tree_label)");
    assert_eq!(out_a.f0_evals_ext, out_b.f0_evals_ext,
        "MergeOutput::f0_evals_ext not deterministic in public inputs");
}

/// C.4 Sanity that the Bundle baseline DOES typically have
/// `f_0_evals_ext` with non-zero entries (catches a degenerate
/// witness silently zero-ing out the reconstruction and making every
/// "diverges" assertion below trivially true).
#[test]
fn c4_baseline_f0_is_nontrivial() {
    let b = build_bundle(0xC004_BEEF);
    let (out, _commit) = run_prover(&b);
    let zero = Ext::from_fp(F::one()) - Ext::from_fp(F::one());
    let nz = out.f0_evals_ext.iter().filter(|&&v| v != zero).count();
    assert!(nz > 0,
        "synthetic Bundle produced trivially-zero f_0; tamper diverges are then meaningless");
}

/// C.5 Tree-label binds the commit: building the SAME witness under
/// two different `tree_label`s produces two different roots, so a
/// reviewer cannot claim "Merkle root collisions across protocols".
#[test]
fn c5_tree_label_binds_commit_root() {
    let b = build_bundle(0xC005_F00D);
    let commit_a = Layer0Commit::from_witness(&b.witness, 0xAAAA_AAAAu64);
    let commit_b = Layer0Commit::from_witness(&b.witness, 0xBBBB_BBBBu64);
    assert_ne!(commit_a.root, commit_b.root,
        "different tree_label values must yield different layer-0 roots");
}

/// C.6 Wrong-root replay rejected by `Layer0Opening::verify_merkle`:
/// an opening built against root_A cannot be verified under root_B.
#[test]
fn c6_opening_against_wrong_root_rejected() {
    let b = build_bundle(0xC006_BEEF);
    let cfg = merkle_cfg(b.h0_domain.len(), b.tree_label);

    let (_out_a, commit_a) = run_prover(&b);

    let mut b2 = b.clone();
    b2.witness.trace_columns[0][0] += F::one();
    let commit_b = Layer0Commit::from_witness(&b2.witness, b2.tree_label);
    assert_ne!(commit_a.root, commit_b.root);

    let opening = commit_a.open(4);
    assert!( opening.verify_merkle(&cfg, commit_a.root));
    assert!(!opening.verify_merkle(&cfg, commit_b.root));
}

// ═══════════════════════════════════════════════════════════════
//  Out-of-scope helper: prevent `HASH_BYTES` warning on unused
// ═══════════════════════════════════════════════════════════════
#[allow(dead_code)]
fn _hash_bytes_in_scope() -> [u8; HASH_BYTES] { [0u8; HASH_BYTES] }
