use rayon;
use ark_ff::UniformRand;
use ark_goldilocks::Goldilocks as F;
use criterion::{
    criterion_group, criterion_main, measurement::WallTime,
    BenchmarkGroup, Criterion, Throughput,
};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

// add to imports
use deep_ali::trace_import::real_trace_inputs;

// ✅ DEEP‑ALI + MF‑FRI APIs
use deep_ali::{
    deep_ali_merge_evals,
    deep_tower::Fp3,
    fri::{
        deep_fri_prove,
        deep_fri_proof_size_bytes,
        deep_fri_verify,
        FriDomain,
        DeepFriParams,
    },
};

// ✅ Extension‑field imports
//use deep_ali::cubic_ext::{CubeExt, GoldilocksCubeConfig};

// ═══════════════════════════════════════════════════════════════════
// Extension‑field selector.  Change this ONE line to switch:
//
//   Fp³ (cubic, default):   type Ext = CubeExt<GoldilocksCubeConfig>;
//   Fp  (base field only):  type Ext = F;
//
// To add Fp⁴ you would need to implement a quartic extension struct
// and implement TowerField for it — not yet available in this crate.
// ═══════════════════════════════════════════════════════════════════
//type Ext = CubeExt<GoldilocksCubeConfig>;

// Build with --features ext-octic for L5 (Fp⁸); default is Fp⁶ (L1/L3).
#[cfg(not(feature = "ext-octic"))]
use deep_ali::sextic_ext::SexticExt;
#[cfg(not(feature = "ext-octic"))]
type Ext = SexticExt;  // Fp⁶ = Fp³[u]/(u² − α)

#[cfg(feature = "ext-octic")]
use deep_ali::octic_ext::OcticExt;
#[cfg(feature = "ext-octic")]
type Ext = OcticExt;   // Fp⁸ for NIST Level 5

// ---------------------
// CSV record
// ---------------------

#[derive(Default, Clone)]
struct CsvRow {
    label: String,
    schedule: String,
    k: usize,
    proof_bytes: usize,
    prove_s: f64,
    verify_ms: f64,
    prove_elems_per_s: f64,
    delta_size_pct: f64,
    delta_prove_pct: f64,
    delta_verify_pct: f64,
    delta_throughput_pct: f64,
}

impl CsvRow {
    fn header() -> &'static str {
        "csv,label,k,schedule,proof_bytes,prove_s,verify_ms,prove_elems_per_s,delta_size_pct_vs_paper,delta_prove_pct_vs_paper,delta_verify_pct_vs_paper,delta_throughput_pct_vs_paper"
    }
    fn to_line(&self) -> String {
        format!(
            "csv,{},{},{},{},{:.6},{:.3},{:.6},{:.2},{:.2},{:.2},{:.2}\n",
            self.label,
            self.k,
            self.schedule,
            self.proof_bytes,
            self.prove_s,
            self.verify_ms,
            self.prove_elems_per_s,
            self.delta_size_pct,
            self.delta_prove_pct,
            self.delta_verify_pct,
            self.delta_throughput_pct
        )
    }
    fn print_stdout(&self) {
        print!("{}", self.to_line());
    }
}

// ---------------------
// Schedule helpers
// ---------------------

fn schedule_str(s: &[usize]) -> String {
    format!(
        "[{}]",
        s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
    )
}

fn log2_pow2(x: usize) -> usize {
    assert!(x.is_power_of_two());
    x.trailing_zeros() as usize
}

fn k_min_for_schedule(schedule: &[usize]) -> usize {
    schedule.iter().map(|&m| log2_pow2(m)).sum()
}

fn divides_chain(n0: usize, schedule: &[usize]) -> bool {
    let mut n = n0;
    for &m in schedule {
        if n % m != 0 {
            return false;
        }
        n /= m;
    }
    true
}

fn ks_for_schedule(schedule: &[usize], k_lo: usize, k_hi: usize) -> Vec<usize> {
    let k_min = k_min_for_schedule(schedule);
    (k_lo.max(k_min)..=k_hi)
        .filter(|&k| divides_chain(1usize << k, schedule))
        .collect()
}

// ✅ Normalize schedule so final layer size = 1
fn normalize_fri_schedule(n0: usize, mut schedule: Vec<usize>) -> Vec<usize> {
    let mut n = n0;
    for &m in &schedule {
        assert!(n % m == 0, "schedule does not divide domain");
        n /= m;
    }
    if n > 1 {
        assert!(n.is_power_of_two(), "final layer must be power of two");
        schedule.push(n);
    }
    schedule
}

// ---------------------
// MF‑FRI benchmark (WITH DEEP‑ALI)
// ---------------------

fn bench_e2e_mf_fri(c: &mut Criterion) {
    eprintln!(
        "[RAYON CHECK] current_num_threads = {}",
        rayon::current_num_threads()
    );

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("e2e_mf_fri");
    g.warm_up_time(Duration::from_secs(5));
    g.measurement_time(Duration::from_secs(20));
    g.sample_size(10);

    // Query count r is per-NIST-level: 54 (L1), 79 (L3), 105 (L5).
    // Override via BENCH_R env var; default 54.
    let r: usize = std::env::var("BENCH_R")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(54);

    let seed_z: u64 = 0xDEEF_BAAD;
    let k_lo = 11usize;
    let k_hi = 25usize;

    let presets: &[(&str, &[usize])] = &[
        ("2power16", &[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]),
        ("4x4x4",    &[4,4,4,4,4,4,4,4]),
        //("884", &[8,8,4]),
        //("paper", &[16, 16, 8]),
        //("mod16", &[16, 16, 16, 16]),
        //("uni32x3", &[32, 32, 32]),
        //("uni64x2x8", &[64, 64, 8]),
        //("hi128_32_8", &[128, 32, 8]),
    ];

    // Output filename overridable via BENCH_CSV_OUT for per-(level,hash,q) runs.
    let out_path = std::env::var("BENCH_CSV_OUT")
        .unwrap_or_else(|_| "benchmarkdata.csv".to_string());
    eprintln!("[OUTPUT] r={}  csv={}", r, out_path);
    let file = File::create(&out_path).unwrap();
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", CsvRow::header()).unwrap();
    println!("{}", CsvRow::header());

    let mut paper_baseline: HashMap<usize, CsvRow> = HashMap::new();
    let mut rng_seed = 1337u64;

    for &(label, schedule) in presets {
        let ks = ks_for_schedule(schedule, k_lo, k_hi);

        eprintln!(
            "\n[PRESET] label={} schedule={} total_k={}",
            label,
            schedule_str(schedule),
            ks.len()
        );

        for &k in &ks {
            let run_start = Instant::now();

            let n0 = 1usize << k;
            g.throughput(Throughput::Elements(n0 as u64));

            let normalized_schedule = normalize_fri_schedule(n0, schedule.to_vec());

            eprintln!(
                "[START] label={} schedule={} k={} (n={})",
                label,
                schedule_str(&normalized_schedule),
                k,
                n0
            );

            rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let mut rng = StdRng::seed_from_u64(rng_seed);

            // ---------------------------------------------------------
            // ✅ REAL TRACE INPUTS (degree-bounded, rate 1/4)
            // ---------------------------------------------------------

            let trace = real_trace_inputs(n0, 4);
            let a_eval = trace.a_eval;
            let s_eval = trace.s_eval;
            let e_eval = trace.e_eval;
            let t_eval = trace.t_eval;

            let domain0 = FriDomain::new_radix2(n0);

            // ✅ DEEP‑ALI merge — current API takes n_trace (rate 1/4 ⇒ n_trace = n0/4)
            // and returns a single Vec<F> (FRI derives z internally via Fiat–Shamir).
            let n_trace = n0 / 4;
            let f0_ali = deep_ali_merge_evals(
                &a_eval,
                &s_eval,
                &e_eval,
                &t_eval,
                domain0.omega,
                n_trace,
            );

            // STIR mode for the 4x4x4 preset; FRI mode (binary) for 2power16.
            let use_stir = label == "4x4x4";
            let params = DeepFriParams {
                schedule: normalized_schedule.clone(),
                r,
                seed_z,
                coeff_commit_final: true,
                d_final: 1,
                stir: use_stir,
                s0: r,
            };

            // ---------------- Prove (using selected Ext) ----------------
            let t0 = Instant::now();
            let proof = deep_fri_prove::<Ext>(f0_ali.clone(), domain0, &params);
            let prove_s = t0.elapsed().as_secs_f64();

            // ---------------- Verify (using selected Ext) ----------------
            let t1 = Instant::now();
            assert!(deep_fri_verify::<Ext>(&params, &proof));
            let verify_ms = t1.elapsed().as_secs_f64() * 1e3;

            let proof_bytes = deep_fri_proof_size_bytes::<Ext>(&proof, params.stir);

            let mut row = CsvRow {
                label: label.to_string(),
                schedule: schedule_str(&normalized_schedule),
                k,
                proof_bytes,
                prove_s,
                verify_ms,
                prove_elems_per_s: n0 as f64 / prove_s,
                delta_size_pct: 0.0,
                delta_prove_pct: 0.0,
                delta_verify_pct: 0.0,
                delta_throughput_pct: 0.0,
            };

            if label == "paper" {
                paper_baseline.insert(k, row.clone());
            } else if let Some(base) = paper_baseline.get(&k) {
                row.delta_size_pct =
                    100.0 * (row.proof_bytes as f64 - base.proof_bytes as f64)
                        / base.proof_bytes as f64;
                row.delta_prove_pct =
                    100.0 * (row.prove_s - base.prove_s) / base.prove_s;
                row.delta_verify_pct =
                    100.0 * (row.verify_ms - base.verify_ms) / base.verify_ms;
                row.delta_throughput_pct =
                    100.0 * (row.prove_elems_per_s - base.prove_elems_per_s)
                        / base.prove_elems_per_s;
            }

            row.print_stdout();
            std::io::stdout().flush().unwrap();
            writer.write_all(row.to_line().as_bytes()).unwrap();
            writer.flush().unwrap();

            eprintln!(
                "[DONE ] label={} k={} prove={:.2}s verify={:.2}ms proof={} bytes  [ext=Fp{}]",
                label,
                k,
                prove_s,
                verify_ms,
                proof_bytes,
                std::mem::size_of::<Ext>() / std::mem::size_of::<F>(),
            );

            let run_secs = run_start.elapsed().as_secs_f64();
            eprintln!(
                "[ETA ] label={} schedule={} elapsed={:.2}s",
                label,
                schedule_str(&normalized_schedule),
                run_secs
            );
        }
    }

    g.finish();
}

criterion_group!(e2e, bench_e2e_mf_fri);
criterion_main!(e2e);