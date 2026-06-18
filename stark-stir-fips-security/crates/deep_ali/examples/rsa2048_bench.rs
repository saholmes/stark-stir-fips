//! RSA-2048 verify STARK measurement.
//!
//! Runs one full prove + verify of the stacked-RSA-2048 verify AIR
//! (single record) and reports prove_ms / verify_ms / proof_kib in a
//! CSV-friendly format.  Intended for aws-bench/bench-rsa2048.sh.
//!
//! Run:
//!     cargo run --release -p deep_ali --example rsa2048_bench \
//!         --features "parallel sha3-256 mldsa-44" --no-default-features

use std::time::Instant;

use ark_ff::Zero;
use ark_goldilocks::Goldilocks as F;
use ark_serialize::{CanonicalSerialize, Compress};
use num_bigint::BigUint;
use rand::{Rng, SeedableRng};

use deep_ali::{
    deep_ali_merge_rsa_stacked_streaming,
    fri::{deep_fri_prove, deep_fri_verify, DeepFriParams, FriDomain},
    rsa2048_stacked_air::{
        build_rsa_stacked_layout, fill_rsa_stacked,
        rsa_stacked_constraints, RsaStackedRecord,
    },
    secured_prove::{
        deep_fri_prove_secured, deep_fri_verify_secured,
        deep_fri_proof_size_bytes_secured, secured_rounds_for,
        SECURED_FOLD_K,
    },
    sextic_ext::SexticExt,
    trace_import::lde_trace_columns,
};

type Ext = SexticExt;

fn gen_biguint(rng: &mut rand::rngs::StdRng, bits: u32) -> BigUint {
    let bytes = (bits as usize + 7) / 8;
    let mut buf = vec![0u8; bytes];
    rng.fill(&mut buf[..]);
    let extra = (bytes * 8) - bits as usize;
    if extra > 0 {
        buf[0] &= 0xFF >> extra;
    }
    BigUint::from_bytes_be(&buf)
}

fn gen_biguint_below(rng: &mut rand::rngs::StdRng, n: &BigUint) -> BigUint {
    let bits = n.bits() as u32;
    loop {
        let candidate = gen_biguint(rng, bits);
        if &candidate < n {
            return candidate;
        }
    }
}

fn main() {
    let rayon_threads = rayon::current_num_threads();
    eprintln!(
        "=== rsa2048_bench: stacked RSA-2048 verify AIR (1 record), rayon_threads={rayon_threads} ==="
    );

    // ── Synthesise one honest RSA verification record ──
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xDEAD);
    let n = (gen_biguint(&mut rng, 2046) << 1) | BigUint::from(1u8);
    let s = gen_biguint_below(&mut rng, &n);
    let em = s.modpow(&BigUint::from(65_537u32), &n);
    let records = vec![RsaStackedRecord { n, s, em }];

    let layout = build_rsa_stacked_layout(records.len());
    // n_trace must accommodate the layout's row schedule; for one
    // record at 2046 bits, ~2080 active rows.  Round up to power-of-2.
    let n_trace_active = 2080usize;
    let n_trace = n_trace_active.next_power_of_two();

    let mut trace: Vec<Vec<F>> = (0..layout.width)
        .map(|_| vec![F::zero(); n_trace]).collect();
    fill_rsa_stacked(&mut trace, &layout, n_trace, &records);

    let blowup: usize = std::env::var("BENCH_BLOWUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let r: usize = std::env::var("BENCH_QUERIES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(54);
    // BENCH_LDT={fri,stir} toggles low-degree test (default: fri).
    let use_stir: bool = matches!(
        std::env::var("BENCH_LDT").as_deref(),
        Ok("stir") | Ok("STIR"),
    );
    let ldt_label = if use_stir { "stir" } else { "fri" };

    let kk = rsa_stacked_constraints(&layout);
    eprintln!(
        "trace cols: {}, rows: {}, constraints: {}, blowup: {}, r: {}, ldt: {}",
        layout.width, n_trace, kk, blowup, r, ldt_label
    );

    // ── Prove ──
    let n0 = n_trace * blowup;
    let domain = FriDomain::new_radix2(n0);
    let pi_hash: [u8; 32] = {
        use sha3::{Digest, Sha3_256};
        let mut h = Sha3_256::new();
        h.update(b"deep_ali/rsa2048_bench/v1");
        h.finalize().into()
    };
    // M1.B — engage the proven Johnson-regime {t_i} schedule when the
    // bench harness sets BENCH_T_SCHEDULE.  The secured path routes to
    // `deep_fri_prove_secured` (per-round-distinct query counts via
    // stir_halve::prove_halve_full_ext, k=4 fold arity), which differs
    // from the uniform-r path in both schedule shape and prover/
    // verifier implementation.  Without BENCH_T_SCHEDULE the bench
    // falls through to the historical uniform-r prover.
    let t_per_round_env: Option<Vec<usize>> = std::env::var("BENCH_T_SCHEDULE")
        .ok()
        .and_then(|raw| {
            let raw = raw.trim().to_string();
            if raw.is_empty() {
                None
            } else {
                Some(
                    raw.split(',')
                        .map(|s| {
                            s.trim()
                                .parse::<usize>()
                                .expect("BENCH_T_SCHEDULE positive int")
                        })
                        .collect(),
                )
            }
        });
    let use_secured = t_per_round_env.is_some();

    // Sweep mode: BENCH_FOLD_K overrides the uniform-mode fold arity
    // (default 2 = FRI-style).  Secured mode uses `secured_fold_k()`,
    // which reads the same env var (default 4).
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
        assert!(
            log2_n0 % log2_k == 0,
            "uniform-mode schedule: log_2(n_0) = {} not divisible by \
             log_2(BENCH_FOLD_K) = {}; pick a compatible BENCH_BLOWUP \
             so n_0 is a power of {}",
            log2_n0, log2_k, bench_fold_k
        );
        (0..log2_n0 / log2_k).map(|_| bench_fold_k).collect()
    };

    let mut params = DeepFriParams {
        schedule: schedule.clone(),
        r,
        seed_z: 0xDEEFu64,
        coeff_commit_final: true,
        d_final: 1,
        stir: use_stir || use_secured,  // secured = STIR-style by construction
        s0: r,
        public_inputs_hash: Some(pi_hash),  // P6.6 — now backported
        t_per_round: None,                  // M1.1 — set below if secured
    };
    if let Some(v) = t_per_round_env.clone() {
        assert_eq!(
            v.len(),
            schedule.len(),
            "BENCH_T_SCHEDULE has {} entries, expected {} (one per secured fold round)",
            v.len(),
            schedule.len()
        );
        eprintln!("secured-schedule: t_per_round = {:?} (k={}, M={})",
                  v, SECURED_FOLD_K, schedule.len());
        params = params.with_t_per_round(v);
    }

    let t0 = Instant::now();
    let lde = lde_trace_columns(&trace, n_trace, blowup).expect("LDE");
    let comb_coeffs: Vec<F> = (0..kk).map(|i| F::from((i + 1) as u64)).collect();
    let (c_eval, _info) = deep_ali_merge_rsa_stacked_streaming(
        &lde, &comb_coeffs, &layout, F::zero(), n_trace, blowup,
    );

    let (proof_bytes, verify_fn): (usize, Box<dyn Fn() -> bool>) = if use_secured {
        let proof = deep_fri_prove_secured::<Ext>(c_eval, domain, &params);
        let bytes = deep_fri_proof_size_bytes_secured::<Ext>(&proof);
        let p_for_verify = proof;
        let params_for_verify = params.clone();
        let n0_for_verify = n0;
        (bytes, Box::new(move || {
            deep_fri_verify_secured::<Ext>(&params_for_verify, &p_for_verify, n0_for_verify)
        }))
    } else {
        let proof = deep_fri_prove::<Ext>(c_eval, domain, &params);
        let bytes = deep_ali::fri::deep_fri_proof_size_bytes::<Ext>(&proof, use_stir);
        let p_for_verify = proof;
        let params_for_verify = params.clone();
        (bytes, Box::new(move || {
            deep_fri_verify::<Ext>(&params_for_verify, &p_for_verify)
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
        assert!(ok, "RSA-2048 verify rejected — bench is broken");
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let verify_ms = samples[1];

    let mode = if use_secured { "secured" } else { "uniform" };
    println!(
        "rsa2048_bench mode={mode} n_trace={n_trace} blowup={blowup} r={r} \
         threads={rayon_threads} \
         prove_ms={prove_ms:.0} verify_ms={verify_ms:.2} proof_kib={proof_kib:.1}"
    );
}
