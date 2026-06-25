//! ECDSA-p256 verify STARK measurement.
//!
//! Two AIR layouts, selected by `BENCH_AIR`:
//!   * `v2` (default) — the v2 single-row verify-air (all chains live
//!     in width, not height).  Full K=256 OOMs on typical hosts at the
//!     flat-row width, so this path is used for the K=2 stub smoke.
//!   * `double_multirow` — the double-chain multi-row AIR (u_1·G and
//!     u_2·Q as two step-gadgets per row, K rows deep) + streaming
//!     merge.  This is the memory-bounded path that runs the FULL
//!     K=256 ECDSA verify in a single STARK without OOM.
//!
//! Scalar bit width via `BENCH_K_SCALAR` (default 256).  Reports
//! prove_ms / verify_ms / proof_kib in the same CSV-friendly format as
//! `rsa2048_bench` and `ed25519_bench` so `bench-all-signatures.sh`
//! can scrape the line.  Honours the paper's STIR-k4 knobs:
//! BENCH_BLOWUP, BENCH_QUERIES, BENCH_FOLD_K, BENCH_LDT, BENCH_T_SCHEDULE.
//!
//! Requires `--features p256-merge-helpers` so the streaming merges
//! are in-tree.

use std::time::Instant;

use ark_ff::{PrimeField, Zero};
use ark_goldilocks::Goldilocks as F;

use deep_ali::{
    deep_ali_merge_ecdsa_double_multirow_streaming,
    deep_ali_merge_p256_ecdsa_v2_streaming,
    fri::{deep_fri_prove, deep_fri_verify, DeepFriParams, FriDomain},
    p256_ecdsa_air_v2::{
        build_ecdsa_verify_v2_layout, ecdsa_verify_v2_constraints, fill_ecdsa_verify_v2,
    },
    p256_ecdsa_double_multirow_air::{
        build_ecdsa_double_multirow_layout, ecdsa_double_multirow_constraints,
        fill_ecdsa_double_multirow,
    },
    p256_field::{FieldElement, NUM_LIMBS},
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

/// Build the filled trace columns for the v2 single-row verify-air.
/// Returns (trace_cols, num_constraints, n_trace).
fn build_v2_trace(k_scalar: usize) -> (Vec<Vec<F>>, usize, usize) {
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

    // v2 is a single-row witness layout (all chains live in width); pad
    // to next-power-of-two of an env-configurable base (default 8).
    let n_trace_active: usize = std::env::var("BENCH_N_TRACE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let n_trace = n_trace_active.next_power_of_two();

    let mut trace_cols: Vec<Vec<F>> = (0..total)
        .map(|_| vec![F::zero(); n_trace]).collect();

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
            let v: u64 = row0_trace[base + i][0].0.0[0];
            limbs[i] = v;
        }
        limbs
    };
    for i in 0..NUM_LIMBS {
        row0_trace[layout.r_input_base + i][0] = F::from(r_x_mod_n_fe[i]);
    }
    for c in 0..total {
        trace_cols[c][0] = row0_trace[c][0];
    }

    let kk = ecdsa_verify_v2_constraints(&layout);
    (trace_cols, kk, n_trace)
}

/// Build the filled trace for the double-chain multi-row AIR at K=k.
/// Two `scalar_mul_step` gadgets per row, k rows deep, with the
/// 2-pass r_proj boundary fill.  Returns
/// (trace_cols, layout, num_constraints, n_trace).
fn build_double_multirow_trace(
    k: usize,
) -> (
    Vec<Vec<F>>,
    deep_ali::p256_ecdsa_double_multirow_air::EcdsaDoubleMultirowLayout,
    usize,
    usize,
) {
    let (layout, total_cells) = build_ecdsa_double_multirow_layout(0);
    let total_constraints = ecdsa_double_multirow_constraints(&layout);
    let n_trace = k.next_power_of_two().max(2);

    let g = *GENERATOR;
    let q = g.double();
    let z_one = {
        let mut t = FieldElement::zero();
        t.limbs[0] = 1;
        t
    };
    let a_bits: Vec<bool> = (0..k).map(|i| i % 2 == 0).collect();
    let b_bits: Vec<bool> = (0..k).map(|i| i % 3 == 0).collect();
    let zero_fe = FieldElement::zero();

    let read_fe = |trace: &[Vec<F>], base: usize, row: usize| -> FieldElement {
        let mut limbs = [0i64; NUM_LIMBS];
        for i in 0..NUM_LIMBS {
            let v = trace[base + i][row];
            let bi = v.into_bigint();
            limbs[i] = bi.as_ref()[0] as i64;
        }
        FieldElement { limbs }
    };

    // Pass 1: capture chain outputs at the last active row.
    let mut trace: Vec<Vec<F>> = (0..total_cells)
        .map(|_| vec![F::zero(); n_trace]).collect();
    fill_ecdsa_double_multirow(
        &mut trace, &layout, n_trace, k, k,
        &g.x, &g.y, &z_one, &g.x, &g.y, &z_one, &a_bits,
        &q.x, &q.y, &z_one, &q.x, &q.y, &z_one, &b_bits,
        &zero_fe, &zero_fe, &zero_fe, &zero_fe, &zero_fe, &zero_fe,
    );
    let last = k - 1;
    let r_a_x = read_fe(&trace, layout.step_a.select_x.c_limbs_base, last);
    let r_a_y = read_fe(&trace, layout.step_a.select_y.c_limbs_base, last);
    let r_a_z = read_fe(&trace, layout.step_a.select_z.c_limbs_base, last);
    let r_b_x = read_fe(&trace, layout.step_b.select_x.c_limbs_base, last);
    let r_b_y = read_fe(&trace, layout.step_b.select_y.c_limbs_base, last);
    let r_b_z = read_fe(&trace, layout.step_b.select_z.c_limbs_base, last);

    // Pass 2: refill with the correct r_proj boundary inputs.
    let mut trace2: Vec<Vec<F>> = (0..total_cells)
        .map(|_| vec![F::zero(); n_trace]).collect();
    fill_ecdsa_double_multirow(
        &mut trace2, &layout, n_trace, k, k,
        &g.x, &g.y, &z_one, &g.x, &g.y, &z_one, &a_bits,
        &q.x, &q.y, &z_one, &q.x, &q.y, &z_one, &b_bits,
        &r_a_x, &r_a_y, &r_a_z, &r_b_x, &r_b_y, &r_b_z,
    );

    (trace2, layout, total_constraints, n_trace)
}

fn main() {
    let rayon_threads = rayon::current_num_threads();
    eprintln!(
        "=== ecdsa_p256_bench: ECDSA-p256 verify (1 signature), rayon_threads={rayon_threads} ==="
    );

    // ── Configurable scalar bit width (default K=256 full ECDSA) ──
    let k_scalar: usize = std::env::var("BENCH_K_SCALAR")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(256);

    // ── AIR layout: "v2" (single-row stub) or "double_multirow" ──
    let air = std::env::var("BENCH_AIR")
        .unwrap_or_else(|_| "v2".to_string());
    let use_double = matches!(air.as_str(), "double_multirow" | "double" | "multirow");

    let blowup: usize = std::env::var("BENCH_BLOWUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let r_q: usize = std::env::var("BENCH_QUERIES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(54);
    let use_stir: bool = matches!(
        std::env::var("BENCH_LDT").as_deref(),
        Ok("stir") | Ok("STIR"),
    );
    let ldt_label = if use_stir { "stir" } else { "fri" };

    // ── Build the filled trace for the selected AIR (untimed) ──
    let t_fill = Instant::now();
    let double_layout;
    let (trace_cols, kk, n_trace): (Vec<Vec<F>>, usize, usize) = if use_double {
        let (tc, layout, kk, n_trace) = build_double_multirow_trace(k_scalar);
        double_layout = Some(layout);
        (tc, kk, n_trace)
    } else {
        double_layout = None;
        build_v2_trace(k_scalar)
    };
    let fill_ms = t_fill.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "air: {}, trace cols: {}, rows: {}, constraints: {}, blowup: {}, r: {}, ldt: {}, k_scalar: {}, fill_ms: {:.0}",
        if use_double { "double_multirow" } else { "v2" },
        trace_cols.len(), n_trace, kk, blowup, r_q, ldt_label, k_scalar, fill_ms,
    );

    // ── Prove ──
    let n0 = n_trace * blowup;
    let domain = FriDomain::new_radix2(n0);
    let pi_hash: [u8; 32] = {
        use sha3::{Digest, Sha3_256};
        let mut h = Sha3_256::new();
        h.update(b"deep_ali/ecdsa_p256_bench/v1");
        h.update(&[k_scalar as u8, blowup as u8, use_double as u8]);
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
        // then one residual fold for the remainder.  Fits any n_0.
        let full = log2_n0 / log2_k;
        let rem = log2_n0 % log2_k;
        let mut s: Vec<usize> = (0..full).map(|_| bench_fold_k).collect();
        if rem > 0 { s.push(1usize << rem); }
        s
    };
    let _ = SECURED_FOLD_K;
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
    let (c_eval, _info) = if use_double {
        let layout = double_layout.as_ref().unwrap();
        deep_ali_merge_ecdsa_double_multirow_streaming(
            &lde, &comb_coeffs, layout, domain.omega, n_trace, blowup,
        )
    } else {
        // v2 single-row layout is rebuilt here cheaply (no fill).
        let g_x = 0; let g_y = NUM_LIMBS; let g_z = 2 * NUM_LIMBS;
        let q_x = 3 * NUM_LIMBS; let q_y = 4 * NUM_LIMBS; let q_z = 5 * NUM_LIMBS;
        let start = 6 * NUM_LIMBS;
        let (layout, _total) = build_ecdsa_verify_v2_layout(
            start, g_x, g_y, g_z, q_x, q_y, q_z, k_scalar,
        );
        deep_ali_merge_p256_ecdsa_v2_streaming(
            &lde, &comb_coeffs, &layout, n_trace, blowup,
        )
    };

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
    let air_tag = if use_double { "double_multirow" } else { "v2" };
    println!(
        "ecdsa_p256_bench mode={mode} air={air_tag} k_scalar={k_scalar} n_trace={n_trace} blowup={blowup} r={r_q} \
         threads={rayon_threads} \
         prove_ms={prove_ms:.0} verify_ms={verify_ms:.2} proof_kib={proof_kib:.1}"
    );
}
