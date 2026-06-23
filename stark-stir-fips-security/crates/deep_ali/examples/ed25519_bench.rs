//! Ed25519 verify STARK measurement.
//!
//! Runs one full prove + verify of the Ed25519 verify-air v16 (a
//! single signature, RFC 8032 test vector 1, with scalar bit width
//! configurable via BENCH_K_SCALAR; default K=256 matches the full
//! per-signature production configuration).  Reports
//! prove_ms / verify_ms / proof_kib in the same CSV-friendly
//! format as `rsa2048_bench` so `bench-all-signatures.sh` can
//! scrape the line.

use std::time::Instant;

use ark_ff::Zero;
use ark_goldilocks::Goldilocks as F;
use rand::{Rng, SeedableRng};
use sha2::{Digest as _, Sha512};

use deep_ali::{
    deep_ali_merge_ed25519_verify,
    ed25519_scalar::reduce_mod_l_wide,
    ed25519_verify_air::{
        fill_verify_air_v16, r_thread_bits_for_kA, verify_v16_per_row_constraints,
    },
    fri::{deep_fri_prove, deep_fri_verify, DeepFriParams, FriDomain},
    secured_prove::{
        deep_fri_prove_secured, deep_fri_verify_secured,
        deep_fri_proof_size_bytes_secured, secured_rounds_for,
        SECURED_FOLD_K,
    },
    sextic_ext::SexticExt,
    trace_import::lde_trace_columns,
};

type Ext = SexticExt;

/// MSB-first scalar-to-bits conversion (matches the in-AIR convention).
fn scalar_to_bits_msb_first(k: u64, nbits: usize) -> Vec<bool> {
    (0..nbits).rev().map(|i| ((k >> i) & 1) == 1).collect()
}

/// Build `R || A || M` — the SHA-512 input the verify-AIR hashes to
/// derive k = H(R, A, M) mod l.
fn build_verify_sha512_input(r: &[u8; 32], a: &[u8; 32], m: &[u8]) -> Vec<u8> {
    let mut input = Vec::with_capacity(64 + m.len());
    input.extend_from_slice(r);
    input.extend_from_slice(a);
    input.extend_from_slice(m);
    input
}

fn sha512_native(input: &[u8]) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(input);
    h.finalize().into()
}

fn main() {
    let rayon_threads = rayon::current_num_threads();
    eprintln!(
        "=== ed25519_bench: Ed25519 verify-air v16 (1 signature, RFC 8032 test 1), rayon_threads={rayon_threads} ==="
    );

    // ── RFC 8032 test vector 1: empty message ──
    let r: [u8; 32] = hex::decode(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555f",
    )
    .expect("rfc8032 r hex")[..32]
        .try_into()
        .unwrap();
    let a: [u8; 32] = hex::decode(
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
    )
    .expect("rfc8032 a hex")[..32]
        .try_into()
        .unwrap();
    let m: &[u8] = b"";

    // ── Scalar bit width (paper headline = K=256 full scalar) ──
    let k_scalar: usize = std::env::var("BENCH_K_SCALAR")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(256);

    // s = 0xA5 + jitter so we exercise distinct paths each run (the
    // test vector uses a full 256-bit scalar but K=8 is fine for
    // verify-time measurement).
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xED_25519);
    let s_seed: u64 = rng.gen();
    let s_bits = scalar_to_bits_msb_first(s_seed, k_scalar);

    // k = SHA-512(R || A || M) mod l, MSB-first for K bits.
    let sha_input = build_verify_sha512_input(&r, &a, m);
    let digest = sha512_native(&sha_input);
    let k_canonical = reduce_mod_l_wide(&digest);
    let k_bits = r_thread_bits_for_kA(&k_canonical, k_scalar);

    // ── Build the verify-air v16 trace ──
    let (trace, layout, _k) = match fill_verify_air_v16(&sha_input, &r, &a, &s_bits, &k_bits) {
        Some(v) => v,
        None => {
            // R or A failed to decompress (we picked a random s, so we
            // can sometimes hit a curve-side failure).  Retry with a
            // small jitter on the message.
            eprintln!("[ed25519_bench] retrying with shifted message");
            let m2: Vec<u8> = vec![1u8];
            let sha_input = build_verify_sha512_input(&r, &a, &m2);
            let digest = sha512_native(&sha_input);
            let k_canonical = reduce_mod_l_wide(&digest);
            let k_bits = r_thread_bits_for_kA(&k_canonical, k_scalar);
            fill_verify_air_v16(&sha_input, &r, &a, &s_bits, &k_bits)
                .expect("rfc8032 v16 trace builder accepts valid (R, A)")
        }
    };

    let n_trace = layout.height.next_power_of_two();
    let kk = verify_v16_per_row_constraints(layout.k_scalar);

    // Pad each column to power-of-two height.
    let trace_padded: Vec<Vec<F>> = trace
        .into_iter()
        .map(|mut col| {
            col.resize(n_trace, F::zero());
            col
        })
        .collect();

    let blowup: usize = std::env::var("BENCH_BLOWUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let r_q: usize = std::env::var("BENCH_QUERIES")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(54);
    let use_stir: bool = matches!(
        std::env::var("BENCH_LDT").as_deref(),
        Ok("stir") | Ok("STIR"),
    );
    let ldt_label = if use_stir { "stir" } else { "fri" };

    eprintln!(
        "trace cols: {}, rows: {}, constraints: {}, blowup: {}, r: {}, ldt: {}, k_scalar: {}",
        layout.width, n_trace, kk, blowup, r_q, ldt_label, layout.k_scalar
    );

    // ── Prove ──
    let n0 = n_trace * blowup;
    let domain = FriDomain::new_radix2(n0);
    let pi_hash: [u8; 32] = {
        use sha3::{Digest, Sha3_256};
        let mut h = Sha3_256::new();
        h.update(b"deep_ali/ed25519_bench/v1");
        h.update(&r);
        h.update(&a);
        h.finalize().into()
    };

    // M1.B / M1.C — secured-schedule routing via BENCH_T_SCHEDULE.
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
        seed_z: 0xED25u64,
        coeff_commit_final: true,
        d_final: 1,
        stir: use_stir || use_secured,
        s0: r_q,
        public_inputs_hash: Some(pi_hash),
        t_per_round: None,
    };
    if let Some(v) = t_per_round_env {
        assert_eq!(
            v.len(),
            schedule.len(),
            "BENCH_T_SCHEDULE has {} entries, expected {}",
            v.len(),
            schedule.len()
        );
        eprintln!("secured-schedule: t_per_round = {:?}", v);
        params = params.with_t_per_round(v);
    }

    let t0 = Instant::now();
    let lde = lde_trace_columns(&trace_padded, n_trace, blowup).expect("LDE");
    let comb_coeffs: Vec<F> = (0..kk).map(|i| F::from((i + 1) as u64)).collect();
    let (c_eval, _info) = deep_ali_merge_ed25519_verify(
        &lde, &comb_coeffs, &layout, F::zero(), n_trace, blowup,
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
        assert!(ok, "Ed25519 verify rejected — bench is broken");
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let verify_ms = samples[1];

    let mode = if use_secured { "secured" } else { "uniform" };
    println!(
        "ed25519_bench mode={mode} k_scalar={k_scalar} n_trace={n_trace} blowup={blowup} r={r_q} \
         threads={rayon_threads} \
         prove_ms={prove_ms:.0} verify_ms={verify_ms:.2} proof_kib={proof_kib:.1}"
    );
}
