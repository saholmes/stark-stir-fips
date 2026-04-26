// benches/simple_recursion.rs
//
// Standalone two-layer recursive STARK benchmark.
//
// Run with:
//   cargo bench --bench simple_recursion
//
// Or as a one-shot test (no Criterion measurement loop):
//   cargo bench --bench simple_recursion -- --test

use criterion::{
    criterion_group, criterion_main, Criterion,
};
use std::io::Write;
use std::time::{Duration, Instant};

use deep_ali::trace_import::real_trace_inputs;
use deep_ali::{
    deep_ali_merge_evals,
    fri::{
        deep_fri_prove,
        deep_fri_proof_size_bytes,
        deep_fri_verify,
        FriDomain,
        DeepFriParams,
    },
};

use deep_ali::sextic_ext::SexticExt;
use deep_ali::tower_field::TowerField;

type Ext = SexticExt;

// ═══════════════════════════════════════════════════════════════════
//  Configuration
// ═══════════════════════════════════════════════════════════════════

const BLOWUP: usize = 32;
const R_QUERIES: usize = 54;
const USE_STIR: bool = true;
const SEED_Z: u64 = 0xDEEF_BAAD;

/// Inner proof log-domain size.  k=20 is fast for development;
/// bump to 24–25 for production-representative numbers.
const INNER_K: usize = 24;

/// FRI fold arity for the inner layer.
const INNER_ARITY: usize = 8;

/// FRI fold arity for the outer layer.
/// Arity-4 cuts the number of FRI rounds in half, which is
/// important because more FRI rounds = more Merkle trees for the
/// verifier AIR to re-hash.
const OUTER_ARITY: usize = 4;

// ═══════════════════════════════════════════════════════════════════
//  Poseidon verifier-cost model
// ═══════════════════════════════════════════════════════════════════

const POSEIDON_T: usize = 12;          // state width
const POSEIDON_RF: usize = 8;          // full rounds
const POSEIDON_RP: usize = 22;         // partial rounds
const POSEIDON_ROUNDS: usize = POSEIDON_RF + POSEIDON_RP; // 30

/// Estimate the total Poseidon permutation calls the STARK verifier
/// executes when checking a proof with the given parameters.
///
/// Covers:
///   • Merkle-path verification (dominates)
///   • Fiat-Shamir transcript challenges
fn estimate_verifier_poseidon_calls(
    inner_k: usize,
    num_queries: usize,
    schedule: &[usize],
    _stir: bool,
) -> usize {
    let num_rounds = schedule.len();
    let mut domain_bits = inner_k;
    let mut total_hashes: usize = 0;

    // Initial LDE commitment: each query opens a Merkle path
    total_hashes += num_queries * domain_bits;

    for i in 0..num_rounds {
        let arity_bits = log2_pow2(schedule[i]);
        domain_bits = domain_bits.saturating_sub(arity_bits);
        if domain_bits == 0 {
            break;
        }
        // Round commitment: queries open paths of depth domain_bits
        total_hashes += num_queries * domain_bits;
    }

    // Fiat-Shamir: ~2 hashes per round for challenge derivation
    total_hashes += 2 * num_rounds;

    total_hashes
}

/// Returns (air_width, num_constraints, padded_trace_rows).
fn verifier_air_dimensions(
    inner_k: usize,
    num_queries: usize,
    schedule: &[usize],
    _stir: bool,
) -> (usize, usize, usize) {
    let n_hashes = estimate_verifier_poseidon_calls(
        inner_k, num_queries, schedule, _stir,
    );
    let n_rows = n_hashes * POSEIDON_ROUNDS;
    let n_rows_padded = n_rows.next_power_of_two();

    let width = POSEIDON_T + 2;       // 14 columns
    let constraints = POSEIDON_T + 2;

    (width, constraints, n_rows_padded)
}

// ═══════════════════════════════════════════════════════════════════
//  Schedule helpers
// ═══════════════════════════════════════════════════════════════════

fn log2_pow2(x: usize) -> usize {
    assert!(x.is_power_of_two(), "{} is not a power of two", x);
    x.trailing_zeros() as usize
}

fn schedule_str(s: &[usize]) -> String {
    format!(
        "[{}]",
        s.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Extend a base schedule so its fold arities multiply to exactly `n0`.
/// Uniform schedules are extended with the same arity; non-uniform
/// schedules get a tail of arity-2 folds.  No single oversized tail
/// fold is ever emitted.
fn normalize_fri_schedule(n0: usize, mut schedule: Vec<usize>) -> Vec<usize> {
    let mut remaining = n0;
    for &m in &schedule {
        assert!(
            remaining % m == 0,
            "schedule entry {} does not divide remaining domain {}",
            m, remaining,
        );
        remaining /= m;
    }
    if remaining <= 1 {
        return schedule;
    }
    assert!(
        remaining.is_power_of_two(),
        "remaining domain {} must be a power of two",
        remaining,
    );

    let remaining_bits = log2_pow2(remaining);

    let is_uniform =
        !schedule.is_empty() && schedule.iter().all(|&x| x == schedule[0]);

    if is_uniform {
        let arity = schedule[0];
        let arity_bits = log2_pow2(arity);
        let full_folds = remaining_bits / arity_bits;
        for _ in 0..full_folds {
            schedule.push(arity);
        }
        let leftover_bits = remaining_bits % arity_bits;
        for _ in 0..leftover_bits {
            schedule.push(2);
        }
    } else {
        for _ in 0..remaining_bits {
            schedule.push(2);
        }
    }

    schedule
}

// ═══════════════════════════════════════════════════════════════════
//  CSV record (self-contained copy for standalone binary)
// ═══════════════════════════════════════════════════════════════════

#[derive(Default, Clone)]
struct CsvRow {
    layer: String,
    air_type: String,
    air_width: usize,
    air_constraints: usize,
    k: usize,
    schedule: String,
    proof_bytes: usize,
    prove_s: f64,
    verify_ms: f64,
    elems_per_s: f64,
}

impl CsvRow {
    fn header() -> &'static str {
        "layer,air_type,air_w,air_c,k,schedule,\
         proof_bytes,prove_s,verify_ms,elems_per_s"
    }
    fn to_line(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{:.6},{:.3},{:.0}\n",
            self.layer,
            self.air_type,
            self.air_width,
            self.air_constraints,
            self.k,
            self.schedule,
            self.proof_bytes,
            self.prove_s,
            self.verify_ms,
            self.elems_per_s,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
//  The benchmark
// ═══════════════════════════════════════════════════════════════════

fn bench_simple_recursion(c: &mut Criterion) {
    eprintln!(
        "\n[CONFIG] BLOWUP={} QUERIES={} STIR={} INNER_K={} \
         INNER_ARITY={} OUTER_ARITY={}",
        BLOWUP, R_QUERIES, USE_STIR, INNER_K, INNER_ARITY, OUTER_ARITY,
    );
    eprintln!(
        "[CONFIG] rho=1/{} delta≈{:.4} bits/query≈{:.2}",
        BLOWUP,
        1.0 - (1.0 / BLOWUP as f64).sqrt(),
        -((1.0 / BLOWUP as f64).sqrt().log2()),
    );

    let mut g = c.benchmark_group("simple_recursion");
    g.warm_up_time(Duration::from_secs(3));
    g.measurement_time(Duration::from_secs(15));
    g.sample_size(10);

    // ─────────────────────────────────────────────────────────────
    //  Layer 0 (inner): Prove a Fibonacci trace
    // ─────────────────────────────────────────────────────────────
    let inner_n0: usize = 1 << INNER_K;
    let inner_n_trace = inner_n0 / BLOWUP;
    let inner_schedule =
        normalize_fri_schedule(inner_n0, vec![INNER_ARITY]);

    let inner_params = DeepFriParams {
        schedule: inner_schedule.clone(),
        r: if USE_STIR { 0 } else { R_QUERIES },
        seed_z: SEED_Z,
        coeff_commit_final: true,
        d_final: 1,
        stir: USE_STIR,
        s0: if USE_STIR { R_QUERIES } else { 0 },
    };

    eprintln!(
        "\n╔══════════════════════════════════════════════════╗"
    );
    eprintln!(
        "║  Layer 0 (inner):  Fibonacci   k={:<3} n_trace={:<8} ║",
        INNER_K, inner_n_trace,
    );
    eprintln!(
        "║  schedule={}  ({} rounds)        ║",
        schedule_str(&inner_schedule),
        inner_schedule.len(),
    );
    eprintln!(
        "╚══════════════════════════════════════════════════╝"
    );

    let inner_trace = real_trace_inputs(inner_n0, BLOWUP);
    let inner_domain = FriDomain::new_radix2(inner_n0);

    let inner_f0 = deep_ali_merge_evals(
        &inner_trace.a_eval,
        &inner_trace.s_eval,
        &inner_trace.e_eval,
        &inner_trace.t_eval,
        inner_domain.omega,
        inner_n_trace,
    );

    // -- prove --
    let t0 = Instant::now();
    let inner_proof =
        deep_fri_prove::<Ext>(inner_f0.clone(), inner_domain, &inner_params);
    let inner_prove_s = t0.elapsed().as_secs_f64();

    // -- verify --
    let tv0 = Instant::now();
    assert!(
        deep_fri_verify::<Ext>(&inner_params, &inner_proof),
        "FAIL: inner proof did not verify",
    );
    let inner_verify_ms = tv0.elapsed().as_secs_f64() * 1e3;

    let inner_bytes =
        deep_fri_proof_size_bytes::<Ext>(&inner_proof, inner_params.stir);

    eprintln!(
        "[L0 DONE] prove={:.3}s  verify={:.2}ms  size={} bytes",
        inner_prove_s, inner_verify_ms, inner_bytes,
    );

    // ─────────────────────────────────────────────────────────────
    //  Estimate outer trace dimensions
    // ─────────────────────────────────────────────────────────────
    let num_queries = R_QUERIES;
    let (ver_w, ver_c, ver_rows) = verifier_air_dimensions(
        INNER_K, num_queries, &inner_schedule, USE_STIR,
    );
    let outer_n_trace = ver_rows;
    let outer_n0 = outer_n_trace * BLOWUP;
    let outer_k = log2_pow2(outer_n0);

    let est_hashes = estimate_verifier_poseidon_calls(
        INNER_K, num_queries, &inner_schedule, USE_STIR,
    );

    eprintln!(
        "\n[COST MODEL] Poseidon calls={} → raw rows={} → padded rows={} \
         → outer n0=2^{} ({})",
        est_hashes,
        est_hashes * POSEIDON_ROUNDS,
        ver_rows,
        outer_k,
        outer_n0,
    );
    eprintln!(
        "[COST MODEL] Verifier AIR: width={} constraints={}",
        ver_w, ver_c,
    );

    // ─────────────────────────────────────────────────────────────
    //  Layer 1 (outer): Prove verifier execution (MOCK)
    // ─────────────────────────────────────────────────────────────
    let outer_schedule =
        normalize_fri_schedule(outer_n0, vec![OUTER_ARITY]);

    let outer_params = DeepFriParams {
        schedule: outer_schedule.clone(),
        r: if USE_STIR { 0 } else { R_QUERIES },
        seed_z: SEED_Z,
        coeff_commit_final: true,
        d_final: 1,
        stir: USE_STIR,
        s0: if USE_STIR { R_QUERIES } else { 0 },
    };

    eprintln!(
        "\n╔══════════════════════════════════════════════════╗"
    );
    eprintln!(
        "║  Layer 1 (outer):  Verifier (mock)  k={:<3}       ║",
        outer_k,
    );
    eprintln!(
        "║  schedule={}  ({} rounds)        ║",
        schedule_str(&outer_schedule),
        outer_schedule.len(),
    );
    eprintln!(
        "╚══════════════════════════════════════════════════╝"
    );

    let outer_trace = real_trace_inputs(outer_n0, BLOWUP);
    let outer_domain = FriDomain::new_radix2(outer_n0);

    let outer_f0 = deep_ali_merge_evals(
        &outer_trace.a_eval,
        &outer_trace.s_eval,
        &outer_trace.e_eval,
        &outer_trace.t_eval,
        outer_domain.omega,
        outer_n_trace,
    );

    // -- prove --
    let t1 = Instant::now();
    let outer_proof =
        deep_fri_prove::<Ext>(outer_f0, outer_domain, &outer_params);
    let outer_prove_s = t1.elapsed().as_secs_f64();

    // -- verify --
    let tv1 = Instant::now();
    assert!(
        deep_fri_verify::<Ext>(&outer_params, &outer_proof),
        "FAIL: outer proof did not verify",
    );
    let outer_verify_ms = tv1.elapsed().as_secs_f64() * 1e3;

    let outer_bytes =
        deep_fri_proof_size_bytes::<Ext>(&outer_proof, outer_params.stir);

    eprintln!(
        "[L1 DONE] prove={:.3}s  verify={:.2}ms  size={} bytes",
        outer_prove_s, outer_verify_ms, outer_bytes,
    );

    // ─────────────────────────────────────────────────────────────
    //  Summary
    // ─────────────────────────────────────────────────────────────
    let total_prove = inner_prove_s + outer_prove_s;
    let compression = inner_bytes as f64 / outer_bytes as f64;
    let overhead = total_prove / inner_prove_s;

    eprintln!("\n┌──────────────────────────────────────────────────────┐");
    eprintln!("│  RECURSION SUMMARY                                   │");
    eprintln!("├──────────────────────────────────────────────────────┤");
    eprintln!(
        "│  Inner (L0): k={:<3}  prove={:>8.3}s   {:>8} bytes  │",
        INNER_K, inner_prove_s, inner_bytes,
    );
    eprintln!(
        "│  Outer (L1): k={:<3}  prove={:>8.3}s   {:>8} bytes  │",
        outer_k, outer_prove_s, outer_bytes,
    );
    eprintln!(
        "│  Total prove:        {:>8.3}s                       │",
        total_prove,
    );
    eprintln!(
        "│  On-chain artifact:  {:>8} bytes   {:.2}ms verify   │",
        outer_bytes, outer_verify_ms,
    );
    eprintln!("├──────────────────────────────────────────────────────┤");
    eprintln!(
        "│  Compression:   {:.2}×  (inner/outer proof size)      │",
        compression,
    );
    eprintln!(
        "│  Overhead:      {:.2}×  (total prove / inner prove)   │",
        overhead,
    );
    eprintln!(
        "│  Extension:     Fp{}                                  │",
        Ext::DEGREE,
    );
    eprintln!("└──────────────────────────────────────────────────────┘");

    // ─────────────────────────────────────────────────────────────
    //  CSV
    // ─────────────────────────────────────────────────────────────
    println!("{}", CsvRow::header());

    let inner_row = CsvRow {
        layer: "L0_inner".into(),
        air_type: "fibonacci".into(),
        air_width: 2,
        air_constraints: 1,
        k: INNER_K,
        schedule: schedule_str(&inner_schedule),
        proof_bytes: inner_bytes,
        prove_s: inner_prove_s,
        verify_ms: inner_verify_ms,
        elems_per_s: inner_n0 as f64 / inner_prove_s,
    };
    print!("{}", inner_row.to_line());

    let outer_row = CsvRow {
        layer: "L1_outer".into(),
        air_type: "poseidon_verifier_mock".into(),
        air_width: ver_w,
        air_constraints: ver_c,
        k: outer_k,
        schedule: schedule_str(&outer_schedule),
        proof_bytes: outer_bytes,
        prove_s: outer_prove_s,
        verify_ms: outer_verify_ms,
        elems_per_s: outer_n0 as f64 / outer_prove_s,
    };
    print!("{}", outer_row.to_line());

    std::io::stdout().flush().unwrap();

    // ─────────────────────────────────────────────────────────────
    //  Criterion bench loop (wraps just the outer proof, since
    //  that's what a recursive verifier would repeatedly invoke)
    // ─────────────────────────────────────────────────────────────
    g.bench_function(
        &format!("outer_prove_k{}", outer_k),
        |b| {
            // Re-build the merged eval outside the timing loop
            let outer_trace_b = real_trace_inputs(outer_n0, BLOWUP);
            let outer_domain_b = FriDomain::new_radix2(outer_n0);
            let outer_f0_b = deep_ali_merge_evals(
                &outer_trace_b.a_eval,
                &outer_trace_b.s_eval,
                &outer_trace_b.e_eval,
                &outer_trace_b.t_eval,
                outer_domain_b.omega,
                outer_n_trace,
            );

            b.iter(|| {
                let p = deep_fri_prove::<Ext>(
                    outer_f0_b.clone(),
                    outer_domain_b,
                    &outer_params,
                );
                assert!(deep_fri_verify::<Ext>(&outer_params, &p));
            });
        },
    );

    g.finish();
}

criterion_group! {
    name = recursion;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(30));
    targets = bench_simple_recursion
}
criterion_main!(recursion);