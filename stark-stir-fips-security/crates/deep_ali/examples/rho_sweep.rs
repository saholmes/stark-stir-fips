//! Rate (ρ₀) sweep: sound STIR-k4 vs sound FRI-k4 through the SAME
//! ÷2-halving merge family, with BOTH baselines correct:
//!   STIR : dom_div=2 (rate improves ρᵢ=ρ₀·(1/2)ⁱ), DECLINING Johnson tᵢ.
//!   FRI  : dom_div=4 (constant rate ρ₀), FLAT Johnson t (the schedule a
//!          sound binary/k-ary FRI actually needs — not STIR's declining
//!          {tᵢ}, which the prior full-ext-fri baseline wrongly borrowed).
//! Both at folding arity k=4. Real proofs, real verification, real sizes.
//! Tests the hypothesis (original STIR Table 2): STIR's argument-size win
//! over FRI GROWS with rate, collapsing toward parity at low rate (ρ=1/32).
//!
//! Run:
//!   cargo run --release -p deep_ali --example rho_sweep --features parallel

use std::time::Instant;
use ark_ff::{Zero, UniformRand};
use ark_goldilocks::Goldilocks as F;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain as Domain};
use rand::{SeedableRng, rngs::StdRng};
use deep_ali::stir_halve::{
    prove_halve, verify_halve, halve_proof_size_bytes, HalveCoset, RoundSchedule,
};

/// Johnson per-query bits at rate 1/rate_inv = ½·log₂(rate_inv);
/// queries to clear λ bits = ⌈λ / bits_per_query⌉.
fn johnson_t(lambda: usize, rate_inv: f64) -> usize {
    let bits_per_q = 0.5 * rate_inv.log2();
    (((lambda as f64) / bits_per_q).ceil() as usize).max(1)
}

fn measure(
    f0: Vec<F>, n0: usize, k: usize, dom_div: usize, t_per_round: &[usize],
) -> (usize, u128, u128, bool) {
    let m = t_per_round.len();
    let domain0 = HalveCoset::root(n0);
    let schedule: Vec<RoundSchedule> =
        (0..m).map(|_| RoundSchedule { deg_div: k, dom_div }).collect();
    let mut rng = StdRng::seed_from_u64(0xA11CE_5EED);
    let alphas: Vec<F> = (0..m).map(|_| F::rand(&mut rng)).collect();
    let q_indices: Vec<Vec<usize>> = (0..m)
        .map(|i| {
            let n_i = n0 / dom_div.pow(i as u32);
            let folded_len = (n_i / k).max(1);
            (0..t_per_round[i])
                .map(|q| (q * 1_299_709 + 0xBEEF_F00D) % folded_len)
                .collect()
        })
        .collect();
    let t = Instant::now();
    let proof = prove_halve(f0, domain0, &alphas, &schedule, &q_indices);
    let prove_ms = t.elapsed().as_millis();
    let bytes = halve_proof_size_bytes(&proof);
    let tv = Instant::now();
    let ok = verify_halve(&proof, &alphas, &schedule);
    let verify_us = tv.elapsed().as_micros();
    (bytes, prove_ms, verify_us, ok)
}

fn main() {
    let lambda: usize = std::env::var("LAMBDA").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let log_d0: usize = std::env::var("LOG_D0").ok().and_then(|s| s.parse().ok()).unwrap_or(18);
    let log_dfinal: usize = std::env::var("LOG_DFINAL").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    let k: usize = 4;
    let log_k = 2; // k=4
    let m = (log_d0 - log_dfinal) / log_k; // degree-folding rounds (÷k each)
    let d0 = 1usize << log_d0;

    println!("# rho-sweep: STIR-k{k} (dom÷2, declining tᵢ) vs FRI-k{k} (dom÷{k}, flat t), λ={lambda}, d=2^{log_d0}, M={m} rounds");
    println!("# rate  | STIR Σt  proof    verify | FRI Σt  proof    verify | FRI/STIR size  verify");
    for &rate_inv in &[4usize, 8, 16, 32] {
        let n0 = d0 * rate_inv;
        let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
        let coeffs: Vec<F> = (0..n0).map(|i| if i < d0 { F::rand(&mut rng) } else { F::zero() }).collect();
        let dom0 = Domain::<F>::new(n0).expect("radix-2 domain");
        let f0 = dom0.fft(&coeffs);

        // STIR: rate improves each round → rate_inv_i = rate_inv · 2^i → declining t.
        let stir_t: Vec<usize> = (0..m)
            .map(|i| johnson_t(lambda, (rate_inv as f64) * (1u64 << i) as f64))
            .collect();
        // FRI: constant rate ρ₀ → flat t.
        let fri_t: Vec<usize> = (0..m).map(|_| johnson_t(lambda, rate_inv as f64)).collect();

        let (sb, sp, sv, sok) = measure(f0.clone(), n0, k, 2, &stir_t);
        let (fb, fp, fv, fok) = measure(f0, n0, k, k, &fri_t);
        let sst: usize = stir_t.iter().sum();
        let fst: usize = fri_t.iter().sum();
        assert!(sok && fok, "both proofs must verify (STIR={sok} FRI={fok})");
        println!(
            "1/{rate_inv:<3} | {sst:>4} {:>6}KiB {sv:>6}us | {fst:>4} {:>6}KiB {fv:>6}us | size={:.2}x verify={:.2}x  (prove {sp}/{fp} ms)",
            sb / 1024, fb / 1024, fb as f64 / sb as f64, fv as f64 / sv as f64,
        );
    }
}
