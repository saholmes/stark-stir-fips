//! ECDSA-p256 verify STARK measurement.
//!
//! Runs one full prove + verify of the ECDSA-p256 verify-air v2
//! (a single signature with the full 256-bit u1, u2 scalars by
//! default; configurable via BENCH_K_SCALAR for K=2 stub smokes).
//! Reports prove_ms / verify_ms / proof_kib in the same
//! CSV-friendly format as `rsa2048_bench` and `ed25519_bench` so
//! `bench-all-signatures.sh` can scrape the line.
//!
//! Requires `--features p256-merge-helpers` so
//! `deep_ali_merge_p256_ecdsa_v2_streaming` is in-tree.

use std::time::Instant;

use ark_ff::Zero;
use ark_goldilocks::Goldilocks as F;

use deep_ali::{
    deep_ali_merge_p256_ecdsa_v2_streaming,
    fri::{deep_fri_prove, deep_fri_verify, DeepFriParams, FriDomain},
    p256_ecdsa_air_v2::{
        build_ecdsa_verify_v2_layout, ecdsa_verify_v2_constraints, fill_ecdsa_verify_v2,
    },
    p256_field::NUM_LIMBS,
    p256_scalar::ScalarElement,
    p256_group::GENERATOR,
    secured_prove::{
        deep_fri_prove_secured, deep_fri_verify_secured,
        deep_fri_proof_size_bytes_secured, secured_rounds_for,
        SECURED_FOLD_K,
    },
    sextic_ext::SexticExt,
    trace_import::lde_trace_columns,
};

type Ext = SexticExt;

fn main() {
    let rayon_threads = rayon::current_num_threads();
    eprintln!(
        "=== ecdsa_p256_bench: ECDSA-p256 verify-air v2 (1 signature), rayon_threads={rayon_threads} ==="
    );

    // ── Configurable scalar bit width (default K=256 full ECDSA) ──
    let k_scalar: usize = std::env::var("BENCH_K_SCALAR")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(256);

    // ── Layout ──
    let g_x = 0;
    let g_y = NUM_LIMBS;
    let g_z = 2 * NUM_LIMBS;
    let q_x = 3 * NUM_LIMBS;
    let q_y = 4 * NUM_LIMBS;
    let q_z = 5 * NUM_LIMBS;
    let start = 6 * NUM_LIMBS;
    let (layout, total) = build_ecdsa_verify_v2_layout(
        start, g_x, g_y, g_z, q_x, q_y, q_z, k_scalar,
    );

    // n_trace must accommodate the layout's row schedule.  v2 is a
    // single-row witness layout (all chains live in width, not
    // height): we pad to next-power-of-two of an env-configurable
    // base.  Default base = 8 matches the in-repo round-trip test;
    // BENCH_N_TRACE allows larger configurations if downstream
    // multirow extensions are wired.
    let n_trace_active: usize = std::env::var("BENCH_N_TRACE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let n_trace = n_trace_active.next_power_of_two();

    let mut trace_cols: Vec<Vec<F>> = (0..total)
        .map(|_| vec![F::zero(); n_trace]).collect();

    // ── Synthesise a verify-able witness ──
    // The bench uses deterministic u1_bits = alternating-1, u2_bits =
    // alternating-0 — that's a valid (deterministic) configuration
    // of the AIR; the verify-air gate is satisfied because the
    // fill function recomputes R = u1·G + u2·Q and pins r = R_x mod n
    // into the boundary check.  Production callers would derive
    // (u1, u2) from a real (z, r, s, Q): u1 = z·s^-1, u2 = r·s^-1.
    let g = *GENERATOR;
    let q_point = g.double();
    let u1_bits: Vec<bool> = (0..k_scalar).map(|i| i % 2 == 0).collect();
    let u2_bits: Vec<bool> = (0..k_scalar).map(|i| i % 3 != 0).collect();

    let mut row0_trace: Vec<Vec<F>> = (0..total)
        .map(|_| vec![F::zero(); 1]).collect();
    let zero_scalar = ScalarElement::zero();
    fill_ecdsa_verify_v2(
        &mut row0_trace, 0, &layout,
        &g.x, &g.y, &q_point.x, &q_point.y,
        &u1_bits, &u2_bits, &zero_scalar,
    );
    let r_x_mod_n_fe = {
        let mut limbs = [0u64; NUM_LIMBS];
        let base = layout.r_x_mod_n_layout.c_limbs_base;
        for i in 0..NUM_LIMBS {
            let v: u64 = row0_trace[base + i][0]
                .0.0[0];  // Goldilocks .0.0 is BigInt limbs[0]
            limbs[i] = v;
        }
        limbs
    };
    for i in 0..NUM_LIMBS {
        row0_trace[layout.r_input_base + i][0] =
            F::from(r_x_mod_n_fe[i]);
    }
    for c in 0..total {
        trace_cols[c][0] = row0_trace[c][0];
    }

    let blowup: usize = std::env::var("BENCH_BLOWUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let r_q: usize = std::env::var("BENCH_QUERIES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(54);
    let use_stir: bool = matches!(
        std::env::var("BENCH_LDT").as_deref(),
        Ok("stir") | Ok("STIR"),
    );
    let ldt_label = if use_stir { "stir" } else { "fri" };

    let kk = ecdsa_verify_v2_constraints(&layout);
    eprintln!(
        "trace cols: {}, rows: {}, constraints: {}, blowup: {}, r: {}, ldt: {}, k_scalar: {}",
        total, n_trace, kk, blowup, r_q, ldt_label, k_scalar
    );

    // ── Prove ──
    let n0 = n_trace * blowup;
    let domain = FriDomain::new_radix2(n0);
    let pi_hash: [u8; 32] = {
        use sha3::{Digest, Sha3_256};
        let mut h = Sha3_256::new();
        h.update(b"deep_ali/ecdsa_p256_bench/v1");
        h.update(&[k_scalar as u8, blowup as u8]);
        h.finalize().into()
    };

    let t_per_round_env: Option<Vec<usize>> = std::env::var("BENCH_T_SCHEDULE")
        .ok()
        .and_then(|raw| {
            let raw = raw.trim().to_string();
            if raw.is_empty() {
                None
            } else {
                Some(raw.split(',')
                    .map(|s| s.trim().parse::<usize>().expect("BENCH_T_SCHEDULE positive int"))
                    .collect())
            }
        });
    let use_secured = t_per_round_env.is_some();
    let bench_fold_k: usize = std::env::var("BENCH_FOLD_K")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(2);
    assert!(
        bench_fold_k >= 2 && bench_fold_k.is_power_of_two(),
        "BENCH_FOLD_K must be a power of two and >= 2; got {}",
        bench_fold_k
    );
    let schedule: Vec<usize> = if use_secured {
        let num_rounds = secured_rounds_for(n0);
        vec![deep_ali::secured_prove::secured_fold_k(); num_rounds]
    } else {
        let log2_n0 = n0.trailing_zeros() as usize;
        let log2_k = bench_fold_k.trailing_zeros() as usize;
        // k-fold with residual: fold by k while >= log2_k bits remain,
        // then one residual fold for the remainder.  Fits any n_0 (like
        // the deployed STIR div-4 schedule's residual), so FRI-k4 runs
        // at the paper's rate 1/32 on every AIR regardless of trace parity.
        let full = log2_n0 / log2_k;
        let rem = log2_n0 % log2_k;
        let mut s: Vec<usize> = (0..full).map(|_| bench_fold_k).collect();
        if rem > 0 { s.push(1usize << rem); }
        s
    };
    let mut params = DeepFriParams {
        schedule: schedule.clone(),
        r: r_q,
        seed_z: 0xEC_D5Au64,
        coeff_commit_final: true,
        d_final: 1,
        stir: use_stir || use_secured,
        s0: r_q,
        public_inputs_hash: Some(pi_hash),
        t_per_round: None,
    };
    if let Some(v) = t_per_round_env {
        assert_eq!(v.len(), schedule.len(),
            "BENCH_T_SCHEDULE has {} entries, expected {}",
            v.len(), schedule.len());
        eprintln!("secured-schedule: t_per_round = {:?}", v);
        params = params.with_t_per_round(v);
    }

    let t0 = Instant::now();
    let lde = lde_trace_columns(&trace_cols, n_trace, blowup).expect("LDE");
    let comb_coeffs: Vec<F> = (0..kk).map(|i| F::from((i + 1) as u64)).collect();
    let (c_eval, _info) = deep_ali_merge_p256_ecdsa_v2_streaming(
        &lde, &comb_coeffs, &layout, n_trace, blowup,
    );

    let (proof_bytes, verify_fn): (usize, Box<dyn Fn() -> bool>) = if use_secured {
        let proof = deep_fri_prove_secured::<Ext>(c_eval, domain, &params);
        let bytes = deep_fri_proof_size_bytes_secured::<Ext>(&proof);
        let params_for_verify = params.clone();
        let n0_for_verify = n0;
        (bytes, Box::new(move || {
            deep_fri_verify_secured::<Ext>(&params_for_verify, &proof, n0_for_verify)
        }))
    } else {
        let proof = deep_fri_prove::<Ext>(c_eval, domain, &params);
        let bytes = deep_ali::fri::deep_fri_proof_size_bytes::<Ext>(&proof, use_stir);
        let params_for_verify = params.clone();
        (bytes, Box::new(move || {
            deep_fri_verify::<Ext>(&params_for_verify, &proof)
        }))
    };
    let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let proof_kib = proof_bytes as f64 / 1024.0;

    // ── Verify (3 runs for median) ──
    let mut samples: Vec<f64> = Vec::with_capacity(3);
    for _ in 0..3 {
        let t0 = Instant::now();
        let ok = verify_fn();
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        assert!(ok, "ECDSA-p256 verify rejected — bench is broken");
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let verify_ms = samples[1];

    let mode = if use_secured { "secured" } else { "uniform" };
    println!(
        "ecdsa_p256_bench mode={mode} k_scalar={k_scalar} n_trace={n_trace} blowup={blowup} r={r_q} \
         threads={rayon_threads} \
         prove_ms={prove_ms:.0} verify_ms={verify_ms:.2} proof_kib={proof_kib:.1}"
    );
}
