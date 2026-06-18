// secured_prove.rs — M1.B bridge
//
// The uniform-r `deep_fri_prove` engages `DeepFriParams::t_per_round`
// only as a Fiat-Shamir-bound upper bound: every round still opens
// `r = max(t_i)` positions even though the proven Johnson-regime
// schedule allows fewer at deeper rounds.  This module wires the
// AIR-composed merge output `f0: Vec<F>` directly to
// `stir_halve::prove_halve_full_ext` so the per-round-distinct
// `{t_i}` engages structurally — the proof actually shrinks at deeper
// rounds.
//
// Inputs match the uniform-r entry point (`f0: Vec<F>`, `domain0:
// FriDomain`, `params: &DeepFriParams`).  When
// `params.t_per_round.is_none()` the caller MUST use `deep_fri_prove`
// instead.

use ark_goldilocks::Goldilocks as F;

use crate::fri::{
    DeepFriParams,
    FriDomain,
    bind_statement_to_transcript_with_pi,
    challenge_ext,
    index_from_seed,
};
use crate::stir_halve::{
    HalveCoset,
    HalveProofFullExt,
    RoundSchedule,
    prove_halve_full_ext,
    verify_halve_full_ext,
};
use crate::tower_field::TowerField;

use transcript::Transcript;

/// Domain separator for the secured-schedule transcript prelude.  The
/// uniform-r prover writes only `DEEP-FRI-T-SCHEDULE-V1` as the bind
/// suffix; the secured prover additionally writes this tag before
/// drawing per-round randomness so the two routes can never produce
/// transcript-compatible proofs.
const SECURED_DS: &[u8] = b"DEEP-FRI-SECURED-PROVE-V1";

/// Default fold arity engaged by the secured-schedule path.  The
/// paper's Theorem 2 schedules at $k = 4$, $M = 8$ at NIST L1; this
/// is the engaged $k$ unless overridden by `BENCH_FOLD_K` (sweep
/// mode, see [`secured_fold_k`]).  The schedule passed via
/// `DeepFriParams::t_per_round` must have length $M$ where
/// $M = \log_k(n_0)$ — enforced by the runtime length check.
pub const SECURED_FOLD_K: usize = 4;

/// Read the secured-schedule fold arity from the environment, falling
/// back to [`SECURED_FOLD_K`] (= 4) when `BENCH_FOLD_K` is unset.
///
/// Used by the arity/blowup sweep (`scripts/sweep-arity-blowup.sh`)
/// to vary $k$ across cells without recompiling.  Panics if the env
/// value is not a power-of-two integer $\ge 2$.
pub fn secured_fold_k() -> usize {
    let k = std::env::var("BENCH_FOLD_K")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(SECURED_FOLD_K);
    assert!(
        k >= 2 && k.is_power_of_two(),
        "BENCH_FOLD_K must be a power of two and >= 2; got {}",
        k
    );
    k
}

/// Build the M-round schedule for the secured ÷2 STIR prover.  Each
/// round folds by `deg_div = k` (from [`secured_fold_k`]) and halves
/// the domain (`dom_div = 2`) per the paper's schedule.
fn secured_round_schedule(num_rounds: usize) -> Vec<RoundSchedule> {
    let k = secured_fold_k();
    (0..num_rounds)
        .map(|_| RoundSchedule { deg_div: k, dom_div: 2 })
        .collect()
}

/// Determine the number of fold rounds implied by an LDE size and the
/// engaged fold arity.  The caller's `params.t_per_round.len()` must
/// equal this value or the prover panics.
pub fn secured_rounds_for(n0: usize) -> usize {
    let k = secured_fold_k();
    assert!(n0.is_power_of_two(), "secured prover requires power-of-two |H_0|");
    let log_n0 = n0.trailing_zeros() as usize;
    let log_k = k.trailing_zeros() as usize;
    assert!(
        log_n0 >= log_k,
        "secured prover requires |H_0| >= k = {}",
        k
    );
    // We stop folding when the round size would drop below k.
    // M = log_k(n_0 / k) = (log2(n_0) - log2(k)) / log2(k)
    // i.e. M rounds bring n_0 down to k.
    (log_n0 - log_k) / log_k + 1
}

/// FS-derive the secured prelude transcript: bind-statement bytes +
/// optional pi_hash + t_per_round + `SECURED_DS` tag.  Both prover
/// and verifier MUST call this so the per-round alphas and query
/// indices reproduce deterministically.
fn secured_prelude<E: TowerField>(
    domain_size: usize,
    params: &DeepFriParams,
) -> Transcript {
    let mut tr = Transcript::new_matching_hash(b"FRI/FS");
    bind_statement_to_transcript_with_pi::<E>(
        &mut tr,
        &params.schedule,
        domain_size,
        params.seed_z,
        params.coeff_commit_final,
        params.stir,
        params.public_inputs_hash,
        params.t_per_round.as_deref(),
    );
    tr.absorb_bytes(SECURED_DS);
    tr
}

/// Derive the M per-round fold randomness values `α_i ∈ E` from the
/// FS prelude.  Distinct per-round labels prevent reuse across
/// rounds.
fn derive_alphas<E: TowerField>(
    tr: &mut Transcript,
    num_rounds: usize,
) -> Vec<E> {
    let mut alphas = Vec::with_capacity(num_rounds);
    for i in 0..num_rounds {
        let mut tag = Vec::with_capacity(16);
        tag.extend_from_slice(b"alpha");
        for byte in i.to_string().bytes() {
            tag.push(byte);
        }
        alphas.push(challenge_ext::<E>(tr, &tag));
    }
    alphas
}

/// Derive the per-round query indices from the FS prelude.  Round $i$
/// receives $t_i$ independent indices into the round-$i$ folded
/// domain (size `domain_size / k^{i+1}`).
fn derive_q_indices(
    tr: &mut Transcript,
    schedule: &[RoundSchedule],
    t_per_round: &[usize],
    domain_size: usize,
) -> Vec<Vec<usize>> {
    assert_eq!(schedule.len(), t_per_round.len());
    let mut q_indices = Vec::with_capacity(schedule.len());
    let mut cur_size = domain_size;
    for (i, (sched, &t_i)) in schedule.iter().zip(t_per_round.iter()).enumerate() {
        let folded_len = cur_size / sched.deg_div;
        let mut round = Vec::with_capacity(t_i);
        for q in 0..t_i {
            let mut tag = Vec::with_capacity(16);
            tag.extend_from_slice(b"qix");
            for byte in i.to_string().bytes() { tag.push(byte); }
            tag.push(b'/');
            for byte in q.to_string().bytes() { tag.push(byte); }
            let seed: F = challenge_ext::<F>(tr, &tag);
            let n_pow2 = folded_len.next_power_of_two();
            let raw = index_from_seed(seed, n_pow2);
            round.push(raw % folded_len);
        }
        q_indices.push(round);
        cur_size = cur_size / sched.dom_div;
    }
    q_indices
}

/// M1.B prover.  Engages the secured ÷2 STIR path at $k = 4$ with
/// per-round-distinct query counts `params.t_per_round`.  Returns a
/// `HalveProofFullExt` whose byte size shrinks with the schedule
/// (deeper rounds open fewer positions).
///
/// Panics if `params.t_per_round.is_none()` — callers that want the
/// uniform-r path must call `deep_fri_prove` directly.
pub fn deep_fri_prove_secured<E: TowerField>(
    f0: Vec<F>,
    domain0: FriDomain,
    params: &DeepFriParams,
) -> HalveProofFullExt<E> {
    let t_per_round = params
        .t_per_round
        .as_ref()
        .expect("deep_fri_prove_secured requires DeepFriParams.t_per_round = Some(_)");
    let n0 = domain0.size;
    let num_rounds = secured_rounds_for(n0);
    assert_eq!(
        t_per_round.len(),
        num_rounds,
        "DeepFriParams.t_per_round.len()={} but secured prover at k={} on |H_0|={} \
         requires len={} rounds",
        t_per_round.len(),
        secured_fold_k(),
        n0,
        num_rounds
    );
    assert!(
        t_per_round.iter().all(|&t| t > 0),
        "DeepFriParams.t_per_round entries must be positive"
    );
    assert_eq!(
        f0.len(),
        n0,
        "f0 length ({}) must match domain0.size ({})",
        f0.len(),
        n0
    );

    let schedule = secured_round_schedule(num_rounds);
    let mut tr = secured_prelude::<E>(n0, params);
    let alphas: Vec<E> = derive_alphas::<E>(&mut tr, num_rounds);
    let q_indices: Vec<Vec<usize>> =
        derive_q_indices(&mut tr, &schedule, t_per_round, n0);

    let halve_coset = HalveCoset::root(n0);
    prove_halve_full_ext::<E>(f0, halve_coset, &alphas, &schedule, &q_indices)
}

/// M1.B verifier.  Re-derives the FS-bound alphas + per-round query
/// indices and checks (a) the underlying ÷2 STIR proof verifies under
/// those alphas, and (b) every query opening in the proof uses the
/// FS-derived `coset_idx`.  Returns `true` iff all checks pass.
pub fn deep_fri_verify_secured<E: TowerField>(
    params: &DeepFriParams,
    proof: &HalveProofFullExt<E>,
    n0: usize,
) -> bool {
    let t_per_round = match params.t_per_round.as_ref() {
        Some(v) => v,
        None => return false,
    };
    let num_rounds = secured_rounds_for(n0);
    if t_per_round.len() != num_rounds {
        return false;
    }
    if proof.queries.len() != num_rounds {
        return false;
    }
    if !t_per_round.iter().all(|&t| t > 0) {
        return false;
    }

    let schedule = secured_round_schedule(num_rounds);
    let mut tr = secured_prelude::<E>(n0, params);
    let alphas: Vec<E> = derive_alphas::<E>(&mut tr, num_rounds);
    let q_indices: Vec<Vec<usize>> =
        derive_q_indices(&mut tr, &schedule, t_per_round, n0);

    // Structural FS-binding: every query.coset_idx must match the
    // FS-derived index.  Without this check, a prover could swap in
    // adversarially-chosen indices and the verifier's per-query
    // openings would still trivially pass since `verify_halve_full_ext`
    // re-verifies the openings against the supplied indices.
    for (i, round) in q_indices.iter().enumerate() {
        if proof.queries[i].len() != round.len() {
            return false;
        }
        for (q, &fs_idx) in round.iter().enumerate() {
            if proof.queries[i][q].coset_idx != fs_idx {
                return false;
            }
        }
    }

    verify_halve_full_ext::<E>(proof, &alphas, &schedule)
}

/// Approximate wire size of a secured-schedule proof in bytes; thin
/// wrapper over `halve_proof_full_ext_size_bytes` so bench callers
/// don't have to import from `stir_halve` directly.
pub fn deep_fri_proof_size_bytes_secured<E: TowerField>(
    proof: &HalveProofFullExt<E>,
) -> usize {
    crate::stir_halve::halve_proof_full_ext_size_bytes::<E>(proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_poly::{
        EvaluationDomain, DenseUVPolynomial, GeneralEvaluationDomain,
        univariate::DensePolynomial,
    };
    use rand::{SeedableRng, rngs::StdRng};
    use crate::cubic_ext::{CubeExt, GoldilocksCubeConfig};
    use crate::fri::DeepFriParams;

    type TestExt = CubeExt<GoldilocksCubeConfig>;

    fn build_low_degree_f0(n: usize, degree: usize, seed: u64) -> Vec<F> {
        let mut rng = StdRng::seed_from_u64(seed);
        let dom = GeneralEvaluationDomain::<F>::new(n).expect("radix-2 domain");
        let poly = DensePolynomial::<F>::rand(degree, &mut rng);
        dom.fft(poly.coeffs())
    }

    /// M1.B round-trip — secured prove + verify under matched params.
    #[test]
    fn secured_round_trip() {
        let n = 1usize << 8;  // 256 = 4^4 = k^M, so M = 4 rounds
        let degree = n / 16 - 1;
        let f0 = build_low_degree_f0(n, degree, 0xB1_BEEF);

        let schedule: Vec<usize> =
            (0..n.trailing_zeros() as usize).map(|_| 2).collect();
        let t_per_round: Vec<usize> = vec![6, 5, 4, 3];
        let domain0 = FriDomain::new_radix2(n);

        let params = DeepFriParams::new(schedule, 6, 0xC0FFEE)
            .with_t_per_round(t_per_round);
        let proof = deep_fri_prove_secured::<TestExt>(f0, domain0, &params);
        assert!(
            deep_fri_verify_secured::<TestExt>(&params, &proof, n),
            "secured round-trip verify failed"
        );
    }

    /// M1.B tamper — swapping a schedule entry on the verifier side
    /// must reject because the FS prelude binds t_per_round.
    #[test]
    fn secured_rejects_t_schedule_tamper() {
        let n = 1usize << 8;
        let degree = n / 16 - 1;
        let f0 = build_low_degree_f0(n, degree, 0xB2_BEEF);

        let schedule: Vec<usize> =
            (0..n.trailing_zeros() as usize).map(|_| 2).collect();
        let domain0 = FriDomain::new_radix2(n);

        let prover_params = DeepFriParams::new(schedule.clone(), 6, 0xC0FFEE)
            .with_t_per_round(vec![6, 5, 4, 3]);
        let proof = deep_fri_prove_secured::<TestExt>(f0, domain0, &prover_params);

        let verifier_params = DeepFriParams::new(schedule, 6, 0xC0FFEE)
            .with_t_per_round(vec![6, 6, 4, 3]); // one entry tampered
        assert!(
            !deep_fri_verify_secured::<TestExt>(&verifier_params, &proof, n),
            "verifier accepted proof under tampered t_per_round"
        );
    }

    /// M1.B tamper — flipping a query.coset_idx must reject because
    /// the verifier independently re-derives indices from FS.
    #[test]
    fn secured_rejects_q_index_tamper() {
        let n = 1usize << 8;
        let degree = n / 16 - 1;
        let f0 = build_low_degree_f0(n, degree, 0xB3_BEEF);

        let schedule: Vec<usize> =
            (0..n.trailing_zeros() as usize).map(|_| 2).collect();
        let domain0 = FriDomain::new_radix2(n);

        let params = DeepFriParams::new(schedule, 6, 0xC0FFEE)
            .with_t_per_round(vec![6, 5, 4, 3]);
        let mut proof = deep_fri_prove_secured::<TestExt>(f0, domain0, &params);

        let orig = proof.queries[0][0].coset_idx;
        // Toggle a distinct in-range index.
        proof.queries[0][0].coset_idx = (orig + 1) % proof.layer_sizes[0].max(1);
        assert!(
            !deep_fri_verify_secured::<TestExt>(&params, &proof, n),
            "verifier accepted proof under tampered coset_idx"
        );
    }

    /// M1.B load-bearing measurement — proof bytes under the
    /// monotone-decreasing schedule are strictly less than under a
    /// uniform-max schedule at the same maximum bound.  This is the
    /// per-round shrinkage the abstract claims.
    #[test]
    fn secured_proof_shrinks_with_schedule() {
        let n = 1usize << 8;
        let degree = n / 16 - 1;
        let f0 = build_low_degree_f0(n, degree, 0xB4_BEEF);

        let schedule: Vec<usize> =
            (0..n.trailing_zeros() as usize).map(|_| 2).collect();
        let domain0 = FriDomain::new_radix2(n);

        // Uniform schedule: r = 6 at every round.
        let uniform = DeepFriParams::new(schedule.clone(), 6, 0xC0FFEE)
            .with_t_per_round(vec![6, 6, 6, 6]);
        let p_uniform =
            deep_fri_prove_secured::<TestExt>(f0.clone(), domain0, &uniform);

        // Secured schedule: same r at round 0, monotone-decreasing
        // afterward — the proven Johnson-regime shape.
        let secured = DeepFriParams::new(schedule, 6, 0xC0FFEE)
            .with_t_per_round(vec![6, 5, 4, 3]);
        let p_secured = deep_fri_prove_secured::<TestExt>(f0, domain0, &secured);

        let size_uniform = deep_fri_proof_size_bytes_secured(&p_uniform);
        let size_secured = deep_fri_proof_size_bytes_secured(&p_secured);
        eprintln!(
            "[M1.B shrinkage] uniform {} B vs secured {} B (-{:.1}%)",
            size_uniform,
            size_secured,
            (1.0 - size_secured as f64 / size_uniform as f64) * 100.0
        );
        assert!(
            size_secured < size_uniform,
            "secured proof ({} B) is not smaller than uniform ({} B) — \
             per-round shrinkage not engaged",
            size_secured,
            size_uniform
        );
    }
}
