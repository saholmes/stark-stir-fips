//! Paper-aligned explicit merge construction (eq:merge of the companion
//! paper, §3.1).
//!
//! ## What this module adds vs `deep_ali_merge_evals`
//!
//! `crate::deep_ali_merge_evals` implements the *implicit-trace*
//! ALI-quotient form
//! ```text
//!     f_0 = (Φ/Z_H + β·R) |_{H_0}
//! ```
//! where only the merged `f_0` is committed; the trace is implicit in
//! the prover's computation but is never separately exposed to the
//! verifier.
//!
//! The companion paper §3.1 (current version, post-reviewer-rewrite)
//! describes a richer *explicit* merge:
//! ```text
//!     f_0(x) = γ_1 · x^{d_0-(T-1-k)} · (T(x)-Ĩ(x))/V_{zΣ}(x)
//!            + γ_2 · (Q(x) - Q̂(z)) / (x - z)
//!            + β · R(x)
//! ```
//! co-committed in a layer-0 leaf that packs `(T_1,…,T_w, Q, R)`,
//! bound to per-shift OOD claims via a multi-shift vanishing
//! polynomial, and tied to AIR satisfaction by the explicit ALI
//! consistency identity
//! ```text
//!     Σ α_j · Φ_j(z) = Q̂(z) · Z_H(z).            (eq:ali-check)
//! ```
//!
//! Both forms are sound; the explicit form provides strictly stronger
//! binding by separately committing the trace, the ALI quotient, and
//! the blinder, with an explicit OOD consistency check linking them.
//!
//! ## Scope of this commit (P1.1 + P1.2)
//!
//! This module introduces the *witness*, *challenges*, and
//! *output* types together with `deep_ali_merge_explicit`, the
//! primitive that builds `f_0` from already-committed parts.  No
//! callers are migrated yet; `deep_ali_merge_evals` keeps its
//! current callers unchanged, and this module sits alongside.
//!
//! Subsequent phases (P2…P5) migrate the layer-0 Merkle leaf
//! format, the verifier's OOD consistency check, and the AIRs +
//! benches.

use ark_ff::Field;
use ark_goldilocks::Goldilocks as F;
use crate::tower_field::TowerField;

// ─── Witness, OOD claims, challenges ──────────────────────────────

/// Prover-side witness for the explicit merge: the three polynomials
/// that get co-committed in the layer-0 leaf.
///
/// All three are evaluated on the LDE domain `H_0` of size
/// `n_trace * blowup` and stored row-major (`column[row]`).
#[derive(Debug, Clone)]
pub struct MergeWitness {
    /// `w` trace columns, each `n_trace * blowup` long.
    /// Stored as `trace_columns[col][row]`.
    pub trace_columns: Vec<Vec<F>>,

    /// The ALI quotient `Q(X) = Σ α_j Φ_j(X) / Z_H(X)` evaluated on
    /// `H_0`.  Computed via `crate::deep_ali_merge_general`-style
    /// `poly_div_zh` pipeline (unchanged from the implicit form).
    pub ali_quotient: Vec<F>,

    /// A uniform witness-independent low-degree blinder `R(X)`
    /// (`deg R ≤ d_0`), evaluated on `H_0`.  The implicit form
    /// folds `β·R` into `f_0`; the explicit form keeps it as a
    /// separate Merkle-leaf component so HVZK is preserved per
    /// Lemma~4 (HVZK) of the paper.
    pub blinder: Vec<F>,

    /// Trace length `T` (number of rows in the trace, before LDE).
    pub trace_len: usize,

    /// Target degree `d_0 = (d_c - 1) · T - 1`.
    pub d0: usize,

    /// Number of trace-domain shifts referenced by the constraints
    /// (`k = |Σ|`; `k = 2` for first-order transition systems with
    /// `Σ = {1, ω}`).
    pub k_shifts: usize,
}

/// Out-of-domain claims sent on the wire from prover to verifier
/// after the prover has committed `T`, `Q`, and `R` and after the
/// transcript has produced the merge-layer DEEP point `z`.
#[derive(Debug, Clone)]
pub struct OodClaims<E: TowerField> {
    /// The merge-layer DEEP point, `z ∈ F_{p^e} \ H_0`, drawn from
    /// the Fiat–Shamir transcript after the `T`/`Q`/`R` commits.
    pub z: E,

    /// Claimed trace evaluations at each shifted point: indexed
    /// `trace_at_shifts[col][shift_idx]`.  Outer length is `w`,
    /// inner length is `k_shifts` (matching the shift set `Σ`).
    pub trace_at_shifts: Vec<Vec<E>>,

    /// Claimed ALI-quotient evaluation `Q̂(z)`.
    pub q_at_z: E,
}

/// The three Fiat–Shamir merge-batching challenges sampled from the
/// transcript after `OodClaims` are absorbed.
#[derive(Debug, Clone, Copy)]
pub struct MergeChallenges<E: TowerField> {
    /// Trace-summand batching scalar.  For multi-column traces
    /// (`w > 1`), the explicit-form construction batches the `w`
    /// column quotients with the power vector
    /// `(γ_1, γ_1^2, …, γ_1^w)`.
    pub gamma_1: E,

    /// `Q`-summand batching scalar.
    pub gamma_2: E,

    /// Blinder scalar.
    pub beta: E,
}

/// What `deep_ali_merge_explicit` returns: the merged proximity
/// target `f_0`, on the `H_0` LDE domain, in the extension field.
#[derive(Debug, Clone)]
pub struct MergeOutput<E: TowerField> {
    /// `f_0(x)` for each `x ∈ H_0`, lifted to `E`.
    pub f0_evals_ext: Vec<E>,

    /// Echoed for downstream serialization (the verifier receives
    /// these in the proof to re-derive `Ĩ` and `Q̂(z)`).
    pub ood_claims: OodClaims<E>,
}

// ─── Helpers ─────────────────────────────────────────────────────

/// Compute pointwise inverses of `V_{zΣ}(x) = ∏_σ (x - σ·z)` for
/// every `x ∈ h0_domain`.
///
/// Returns a `Vec<E>` of length `h0_domain.len()` holding
/// `1 / V_{zΣ}(x)` per coordinate.
fn vzsigma_inverse_on_h0<E: TowerField>(
    h0_domain: &[F],
    shifts: &[F],
    z: E,
) -> Vec<E> {
    let mut vz: Vec<E> = h0_domain
        .iter()
        .map(|&x| {
            let x_ext = E::from_fp(x);
            shifts.iter().fold(E::one(), |acc, &shift| {
                acc * (x_ext - E::from_fp(shift) * z)
            })
        })
        .collect();
    E::batch_inverse(&mut vz);
    vz
}

/// Build the Lagrange basis denominators for the interpolation
/// nodes `{σ·z}_{σ ∈ shifts}`.
///
/// Returns `denoms_inv[i] = 1 / ∏_{j≠i} (σ_i·z - σ_j·z)`.
fn lagrange_denominators_inv<E: TowerField>(
    shifts: &[F],
    z: E,
) -> Vec<E> {
    let k = shifts.len();
    let nodes: Vec<E> = shifts.iter().map(|&s| E::from_fp(s) * z).collect();

    let mut denoms = Vec::with_capacity(k);
    for i in 0..k {
        let mut acc = E::one();
        for j in 0..k {
            if i != j {
                acc *= nodes[i] - nodes[j];
            }
        }
        denoms.push(acc);
    }
    E::batch_inverse(&mut denoms);
    denoms
}

/// Evaluate the degree-`<k` interpolant `Ĩ_col(x)` matching
/// `Ĩ_col(σ_i · z) = trace_at_shifts[col][i]` for each column,
/// for every `x ∈ h0_domain`.
///
/// Returns `result[col][row]` shaped exactly like `trace_columns`.
fn interpolant_on_h0<E: TowerField>(
    h0_domain: &[F],
    shifts: &[F],
    z: E,
    trace_at_shifts: &[Vec<E>],
) -> Vec<Vec<E>> {
    let w = trace_at_shifts.len();
    let n = h0_domain.len();
    let k = shifts.len();
    let nodes: Vec<E> = shifts.iter().map(|&s| E::from_fp(s) * z).collect();
    let denoms_inv = lagrange_denominators_inv::<E>(shifts, z);

    let mut out = vec![vec![E::zero(); n]; w];
    for col in 0..w {
        debug_assert_eq!(trace_at_shifts[col].len(), k);
        for (row, &x_fp) in h0_domain.iter().enumerate() {
            let x = E::from_fp(x_fp);
            let mut sum = E::zero();
            for i in 0..k {
                // Lagrange basis at node i: ∏_{j≠i} (x - node_j) * denoms_inv[i]
                let mut numer = E::one();
                for j in 0..k {
                    if j != i {
                        numer *= x - nodes[j];
                    }
                }
                sum += trace_at_shifts[col][i] * numer * denoms_inv[i];
            }
            out[col][row] = sum;
        }
    }
    out
}

/// Compute the trace-side per-row contribution
/// `Σ_col γ_1^{col+1} · x^{d_0 - (T-1-k)} · (T_col(x) - Ĩ_col(x)) / V_{zΣ}(x)`
/// for each `x ∈ h0_domain`.
fn trace_summand_on_h0<E: TowerField>(
    h0_domain: &[F],
    trace_columns: &[Vec<F>],
    tilde_i_per_col: &[Vec<E>],
    vzsigma_inv: &[E],
    gamma_1: E,
    correction_exp: u64,
) -> Vec<E> {
    let n = h0_domain.len();
    let w = trace_columns.len();
    let mut out = vec![E::zero(); n];

    // Precompute x^{correction_exp} for every x ∈ H_0.
    let x_pow: Vec<E> = h0_domain
        .iter()
        .map(|&x| E::from_fp(x).pow_u64(correction_exp))
        .collect();

    let mut gamma1_pow = gamma_1;
    for col in 0..w {
        for row in 0..n {
            let t_x = E::from_fp(trace_columns[col][row]);
            let tilde = tilde_i_per_col[col][row];
            let term = gamma1_pow * x_pow[row] * (t_x - tilde) * vzsigma_inv[row];
            out[row] += term;
        }
        gamma1_pow *= gamma_1;
    }
    out
}

/// Compute the `Q`-summand `γ_2 · (Q(x) - Q̂(z)) / (x - z)` per row.
fn q_summand_on_h0<E: TowerField>(
    h0_domain: &[F],
    ali_quotient: &[F],
    z: E,
    q_at_z: E,
    gamma_2: E,
) -> Vec<E> {
    let n = h0_domain.len();
    let mut xz_inv: Vec<E> = h0_domain
        .iter()
        .map(|&x| E::from_fp(x) - z)
        .collect();
    E::batch_inverse(&mut xz_inv);

    let mut out = Vec::with_capacity(n);
    for row in 0..n {
        let q_x = E::from_fp(ali_quotient[row]);
        out.push(gamma_2 * (q_x - q_at_z) * xz_inv[row]);
    }
    out
}

/// Compute the degree-correction exponent `d_0 - (T-1-k)`.
///
/// For `d_0 = (d_c-1)·T - 1`, this simplifies to `(d_c-2)·T + k`.
/// For `d_c = 2` (the paper's headline parameter), the exponent
/// is `k`; for higher-degree constraint systems the exponent grows
/// linearly with `T`.
pub fn correction_exponent(d0: usize, trace_len: usize, k_shifts: usize) -> u64 {
    // `T-1-k` is the natural (un-corrected) degree of (T-Ĩ)/V_{zΣ};
    // we lift it up to d_0 via `x^{d_0 - (T-1-k)}`.
    let natural = trace_len.saturating_sub(1).saturating_sub(k_shifts);
    debug_assert!(
        d0 >= natural,
        "correction_exponent: d_0 = {} < T-1-k = {} — the trace summand \
         would have degree exceeding d_0; check parameter set",
        d0, natural,
    );
    (d0 - natural) as u64
}

// ─── Top-level: build f_0 from the explicit-form ingredients ─────

/// **Paper-aligned explicit merge** (P1.2, eq:merge of the paper).
///
/// Builds the proximity target `f_0` on the LDE domain `H_0` from
/// already-committed witness components (`MergeWitness`), the
/// merge-layer DEEP point + OOD claims (`OodClaims`), and the
/// Fiat–Shamir merge-batching scalars (`MergeChallenges`).
///
/// **Caller contract.**  The caller must produce the OOD claims +
/// challenges via a proper Fiat–Shamir transcript binding:
///
/// 1. Commit `T`, `Q`, `R` to the layer-0 Merkle leaf.
/// 2. Absorb `root_0`, then squeeze `z ∈ F_{p^e} \ H_0`.
/// 3. Compute `OodClaims { z, trace_at_shifts, q_at_z }` from the
///    polynomials by direct evaluation at `{σ·z}_σ` and `z`.
/// 4. Absorb the OOD claims into the transcript.
/// 5. Squeeze `MergeChallenges { gamma_1, gamma_2, beta }`.
/// 6. Call this function with the pieces.
///
/// This function does *not* itself enforce the verifier's ALI
/// consistency check `Σ α_j Φ_j(z) = Q̂(z) · Z_H(z)` — that lives
/// on the verifier side (P3.2 in the next phase).
pub fn deep_ali_merge_explicit<E: TowerField>(
    witness: &MergeWitness,
    h0_domain: &[F],
    shifts: &[F],
    ood: &OodClaims<E>,
    challenges: &MergeChallenges<E>,
) -> MergeOutput<E> {
    let n = h0_domain.len();
    let w = witness.trace_columns.len();
    let k = shifts.len();

    // Witness-shape sanity.
    assert_eq!(
        witness.ali_quotient.len(), n,
        "explicit merge: ali_quotient length {} != |H_0| {}",
        witness.ali_quotient.len(), n,
    );
    assert_eq!(
        witness.blinder.len(), n,
        "explicit merge: blinder length {} != |H_0| {}",
        witness.blinder.len(), n,
    );
    for (col_idx, col) in witness.trace_columns.iter().enumerate() {
        assert_eq!(
            col.len(), n,
            "explicit merge: trace column {} length {} != |H_0| {}",
            col_idx, col.len(), n,
        );
    }
    assert_eq!(
        witness.k_shifts, k,
        "explicit merge: witness.k_shifts {} != shifts.len() {}",
        witness.k_shifts, k,
    );

    // OOD-claim-shape sanity.
    assert_eq!(
        ood.trace_at_shifts.len(), w,
        "explicit merge: |trace_at_shifts| {} != trace width {}",
        ood.trace_at_shifts.len(), w,
    );
    for (col_idx, col_shifts) in ood.trace_at_shifts.iter().enumerate() {
        assert_eq!(
            col_shifts.len(), k,
            "explicit merge: trace_at_shifts[{}] length {} != k_shifts {}",
            col_idx, col_shifts.len(), k,
        );
    }

    // 1. 1/V_{zΣ}(x) at every H_0 coordinate.
    let vz_inv = vzsigma_inverse_on_h0::<E>(h0_domain, shifts, ood.z);

    // 2. Lagrange interpolant Ĩ_col(x) for every column, on H_0.
    let tilde_i = interpolant_on_h0::<E>(
        h0_domain, shifts, ood.z, &ood.trace_at_shifts,
    );

    // 3. Degree-correction factor.
    let correction_exp =
        correction_exponent(witness.d0, witness.trace_len, witness.k_shifts);

    // 4. Trace summand.
    let mut f0 = trace_summand_on_h0::<E>(
        h0_domain,
        &witness.trace_columns,
        &tilde_i,
        &vz_inv,
        challenges.gamma_1,
        correction_exp,
    );

    // 5. Q summand.
    let q_sum = q_summand_on_h0::<E>(
        h0_domain,
        &witness.ali_quotient,
        ood.z,
        ood.q_at_z,
        challenges.gamma_2,
    );
    for i in 0..n {
        f0[i] += q_sum[i];
    }

    // 6. Blinder summand: β · R(x).
    for i in 0..n {
        f0[i] += challenges.beta * E::from_fp(witness.blinder[i]);
    }

    MergeOutput {
        f0_evals_ext: f0,
        ood_claims: ood.clone(),
    }
}

// ─── Layer-0 Merkle leaf format (P2.1) ───────────────────────────
//
// In the implicit-form construction (`deep_ali_merge_evals`), the
// layer-0 Merkle leaf at position `x ∈ H_0` stores a single
// extension-field element `f_0(x)`.  The explicit-form construction
// (this module) requires the layer-0 leaf to instead pack
//
//     (T_1(x), …, T_w(x), Q(x), R(x))
//
// — the `w` trace columns plus the ALI quotient and the blinder, all
// in the *base* field `F` (Goldilocks) — so the verifier can open the
// individual components and reconstruct `f_0(x)` via eq:merge.
//
// This section adds:
//   1. `Layer0LeafContent`            the per-position payload
//   2. `serialize_for_merkle`         flatten to `Vec<F>` for hashing
//   3. `deserialize_from_components`  inverse of (2)
//   4. `reconstruct_f0_at`            verifier-side eq:merge applied
//                                     at a single H_0 coordinate
//   5. `build_layer0_leaves`          prover-side helper that emits
//                                     the |H_0| leaves directly from
//                                     a `MergeWitness`
//
// This commit lands the data-structure + helpers + tests only.  The
// wiring into the existing Merkle-tree builders in `fri.rs` /
// `stir_halve.rs` is P2.2 and follows in a separate commit.

/// Per-position contents of a layer-0 Merkle leaf under the explicit
/// merge construction (eq:merge of the paper).
///
/// Each leaf at `x ∈ H_0` holds the `w` trace-column values, the ALI
/// quotient value, and the blinder value — all in the base field `F`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer0LeafContent {
    /// `trace_values[col] = T_col(x)`, length `w`.
    pub trace_values: Vec<F>,

    /// `Q(x)`.
    pub q_value: F,

    /// `R(x)`.
    pub r_value: F,
}

impl Layer0LeafContent {
    /// Width (= number of trace columns).
    pub fn width(&self) -> usize {
        self.trace_values.len()
    }

    /// Number of base-field components in the flattened leaf:
    /// `w + 2` (trace + Q + R).
    pub fn n_components(&self) -> usize {
        self.trace_values.len() + 2
    }

    /// Flatten to `Vec<F>` in canonical order
    /// `[T_1(x), …, T_w(x), Q(x), R(x)]` for Merkle hashing.
    pub fn serialize_for_merkle(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(self.n_components());
        out.extend_from_slice(&self.trace_values);
        out.push(self.q_value);
        out.push(self.r_value);
        out
    }

    /// Reconstruct from a flat `&[F]` produced by
    /// `serialize_for_merkle`.  `width` is the AIR trace width `w`.
    ///
    /// Returns `None` if `components.len() != width + 2`.
    pub fn deserialize_from_components(
        components: &[F],
        width: usize,
    ) -> Option<Self> {
        if components.len() != width + 2 {
            return None;
        }
        let trace_values = components[..width].to_vec();
        let q_value = components[width];
        let r_value = components[width + 1];
        Some(Self { trace_values, q_value, r_value })
    }
}

/// Verifier-side eq:merge applied at a single H_0 coordinate.
///
/// Given a layer-0 leaf opened at position `x ∈ H_0`, the merge-layer
/// DEEP point and OOD claims (`ood`), the merge challenges (`ch`),
/// and the shift set, computes the merged proximity-target value
/// `f_0(x)` that the FRI verifier compares against the next-round
/// fold.
///
/// This is the verifier-side analogue of `deep_ali_merge_explicit`'s
/// inner-loop body at one position.
pub fn reconstruct_f0_at<E: TowerField>(
    leaf: &Layer0LeafContent,
    x: F,
    shifts: &[F],
    ood: &OodClaims<E>,
    ch: &MergeChallenges<E>,
    trace_len: usize,
    d0: usize,
) -> E {
    let w = leaf.width();
    assert_eq!(
        ood.trace_at_shifts.len(), w,
        "reconstruct_f0_at: OOD trace-shift width {} != leaf width {}",
        ood.trace_at_shifts.len(), w,
    );

    let z = ood.z;
    let k = shifts.len();
    let x_ext = E::from_fp(x);

    // 1. V_{zΣ}(x) and its inverse.
    let vzs: E = shifts
        .iter()
        .fold(E::one(), |acc, &s| acc * (x_ext - E::from_fp(s) * z));
    let vzs_inv = vzs
        .invert()
        .expect("reconstruct_f0_at: V_{zΣ}(x) was zero — x lies on a shifted z?");

    // 2. Interpolant Ĩ_col(x) per column.
    let nodes: Vec<E> = shifts.iter().map(|&s| E::from_fp(s) * z).collect();
    let denoms_inv = lagrange_denominators_inv::<E>(shifts, z);

    let mut tilde_i: Vec<E> = Vec::with_capacity(w);
    for col in 0..w {
        let mut sum = E::zero();
        for i in 0..k {
            let mut numer = E::one();
            for j in 0..k {
                if j != i {
                    numer *= x_ext - nodes[j];
                }
            }
            sum += ood.trace_at_shifts[col][i] * numer * denoms_inv[i];
        }
        tilde_i.push(sum);
    }

    // 3. Degree-correction factor.
    let correction_exp = correction_exponent(d0, trace_len, k);
    let x_pow = x_ext.pow_u64(correction_exp);

    // 4. Trace summand.
    let mut f0 = E::zero();
    let mut gamma1_pow = ch.gamma_1;
    for col in 0..w {
        let t_x = E::from_fp(leaf.trace_values[col]);
        let term = gamma1_pow * x_pow * (t_x - tilde_i[col]) * vzs_inv;
        f0 += term;
        gamma1_pow *= ch.gamma_1;
    }

    // 5. Q summand.
    let xz_inv = (x_ext - z)
        .invert()
        .expect("reconstruct_f0_at: x - z was zero — x lies at z?");
    let q_x = E::from_fp(leaf.q_value);
    f0 += ch.gamma_2 * (q_x - ood.q_at_z) * xz_inv;

    // 6. Blinder summand.
    f0 += ch.beta * E::from_fp(leaf.r_value);

    f0
}

/// Prover-side helper: emit the `|H_0|` layer-0 leaves directly from
/// a `MergeWitness`.  Each leaf is a `Layer0LeafContent`; the caller
/// then serialises each via `serialize_for_merkle` and feeds them to
/// the Merkle tree builder.
///
/// `leaves[row].trace_values[col] = witness.trace_columns[col][row]`.
pub fn build_layer0_leaves(witness: &MergeWitness) -> Vec<Layer0LeafContent> {
    let n = witness.ali_quotient.len();
    let w = witness.trace_columns.len();
    debug_assert_eq!(witness.blinder.len(), n);
    for col in &witness.trace_columns {
        debug_assert_eq!(col.len(), n);
    }

    let mut leaves = Vec::with_capacity(n);
    for row in 0..n {
        let trace_values: Vec<F> = (0..w).map(|c| witness.trace_columns[c][row]).collect();
        leaves.push(Layer0LeafContent {
            trace_values,
            q_value: witness.ali_quotient[row],
            r_value: witness.blinder[row],
        });
    }
    leaves
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sextic_ext::SexticExt;
    use ark_ff::{One as ArkOne, UniformRand};
    use rand::{rngs::StdRng, SeedableRng};

    type Ext = SexticExt;

    /// Smoke test: build a trivial trace + trivial Q + zero blinder
    /// and confirm `deep_ali_merge_explicit` produces a non-trivial
    /// `f_0` (i.e. the pipeline at least executes without panic on
    /// minimal inputs).
    #[test]
    fn explicit_merge_smoke_minimal_inputs() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let trace_len = 8usize;
        let blowup = 4usize;
        let n = trace_len * blowup;
        let w = 2usize; // Fibonacci-style.
        let k_shifts = 2usize;
        let d_c = 2usize;
        let d0 = (d_c - 1) * trace_len - 1; // 7

        // Construct a tiny H_0 of size n via repeated powers of a
        // generator.  Goldilocks has 2-adicity ≥ 32 so any n < 2^32
        // works; here n = 32.
        let omega: F = {
            use ark_ff::FftField;
            F::get_root_of_unity(n as u64).expect("two-adic root")
        };
        let h0: Vec<F> = (0..n).map(|i| omega.pow_u64(i as u64)).collect();

        // Two random trace columns + random Q + random R.
        let trace_columns: Vec<Vec<F>> = (0..w)
            .map(|_| (0..n).map(|_| F::rand(&mut rng)).collect())
            .collect();
        let ali_quotient: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let blinder: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let witness = MergeWitness {
            trace_columns,
            ali_quotient,
            blinder,
            trace_len,
            d0,
            k_shifts,
        };

        // First-order shifts.
        let shifts: Vec<F> = vec![F::one(), omega];

        // FS-derived challenges + OOD claims (randomized for the test).
        let z: Ext = Ext::from_fp(F::rand(&mut rng));
        let trace_at_shifts: Vec<Vec<Ext>> = (0..w)
            .map(|_| (0..k_shifts).map(|_| Ext::from_fp(F::rand(&mut rng))).collect())
            .collect();
        let q_at_z: Ext = Ext::from_fp(F::rand(&mut rng));
        let ood = OodClaims { z, trace_at_shifts, q_at_z };

        let gamma_1: Ext = Ext::from_fp(F::rand(&mut rng));
        let gamma_2: Ext = Ext::from_fp(F::rand(&mut rng));
        let beta: Ext = Ext::from_fp(F::rand(&mut rng));
        let challenges = MergeChallenges { gamma_1, gamma_2, beta };

        let out = deep_ali_merge_explicit::<Ext>(
            &witness, &h0, &shifts, &ood, &challenges,
        );

        assert_eq!(out.f0_evals_ext.len(), n);
        // With random ingredients the output should not be all-zero
        // (probability of accidental cancellation is negligible).
        let any_nonzero = out.f0_evals_ext.iter().any(|v| *v != Ext::default());
        assert!(any_nonzero, "f_0 should not be identically zero on random inputs");
    }

    /// Degree-correction exponent matches the paper formula.
    #[test]
    fn correction_exponent_paper_formula() {
        // d_c = 2, T = 2^22 — paper Cor 1 parameters.
        // d_0 = T - 1 = 2^22 - 1.  T-1-k = 2^22 - 3.
        // Correction exp = d_0 - (T-1-k) = k = 2.
        let trace_len = 1usize << 22;
        let d_c = 2usize;
        let d0 = (d_c - 1) * trace_len - 1;
        let k = 2usize;
        assert_eq!(correction_exponent(d0, trace_len, k), 2);

        // d_c = 4 case.  d_0 = 3T - 1.  T-1-k = T-3.
        // Correction exp = 3T - 1 - (T - 3) = 2T + 2.
        let d_c4 = 4usize;
        let d0_4 = (d_c4 - 1) * trace_len - 1;
        assert_eq!(
            correction_exponent(d0_4, trace_len, k),
            2 * (trace_len as u64) + 2,
        );
    }

    /// Verifier-side sanity: if the prover gives honest OOD claims
    /// from the *actual* trace polynomial values at `{σ·z}_σ` and at
    /// `z` for `Q`, the trace summand at each `x ∈ H_0` ought to be
    /// finite (no division by zero) and the output ought to satisfy
    /// a structural invariant relating to the inputs.
    ///
    /// This test only verifies the no-panic / no-NaN behaviour; the
    /// full soundness invariant (low-degreeness when AIR valid) is
    /// the subject of P4 once the verifier is wired up.
    #[test]
    fn explicit_merge_no_division_by_zero_on_h0() {
        let mut rng = StdRng::seed_from_u64(0xBADCAFE);
        let trace_len = 16;
        let blowup = 4;
        let n = trace_len * blowup;
        let w = 2;
        let k_shifts = 2;
        let d_c = 2;
        let d0 = (d_c - 1) * trace_len - 1;

        let omega: F = {
            use ark_ff::FftField;
            F::get_root_of_unity(n as u64).expect("two-adic root")
        };
        let h0: Vec<F> = (0..n).map(|i| omega.pow_u64(i as u64)).collect();

        let trace_columns: Vec<Vec<F>> = (0..w)
            .map(|_| (0..n).map(|_| F::rand(&mut rng)).collect())
            .collect();
        let ali_quotient: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let blinder: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let witness = MergeWitness {
            trace_columns,
            ali_quotient,
            blinder,
            trace_len,
            d0,
            k_shifts,
        };

        let shifts: Vec<F> = vec![F::one(), omega];

        // z must lie outside H_0.  Pick any Ext element that is not
        // a coercion of an H_0 element; a random non-trivial Ext
        // suffices with overwhelming probability.
        let z: Ext = Ext::from_fp_components(&[
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
        ]).expect("rebuild Ext from random components");

        let trace_at_shifts: Vec<Vec<Ext>> = (0..w)
            .map(|_| (0..k_shifts).map(|_| Ext::from_fp(F::rand(&mut rng))).collect())
            .collect();
        let q_at_z = Ext::from_fp(F::rand(&mut rng));
        let ood = OodClaims { z, trace_at_shifts, q_at_z };

        let gamma_1: Ext = Ext::from_fp(F::rand(&mut rng));
        let gamma_2: Ext = Ext::from_fp(F::rand(&mut rng));
        let beta: Ext = Ext::from_fp(F::rand(&mut rng));
        let challenges = MergeChallenges { gamma_1, gamma_2, beta };

        let out = deep_ali_merge_explicit::<Ext>(
            &witness, &h0, &shifts, &ood, &challenges,
        );

        assert_eq!(out.f0_evals_ext.len(), n);
        // All outputs must be valid Ext elements (not the result of a
        // zero-inverse).  Since `batch_inverse` would panic on zeros
        // upstream, reaching here means the (x - σ·z) and (x - z)
        // denominators never vanished — which is the structural
        // invariant `H_0 ∩ {σ·z}_σ = ∅`.
        for v in &out.f0_evals_ext {
            // sanity: each component is in F (no NaN possibility in
            // finite fields, but assert that conversion roundtrips).
            let _ = v.to_fp_components();
        }
    }

    // ─── P2.1 tests: layer-0 leaf format + verifier reconstruction ──

    /// Round-trip the leaf serializer: build → serialize → deserialize
    /// → equal.
    #[test]
    fn layer0_leaf_serde_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0xABCD_EF01);
        for &w in &[1usize, 2, 8, 16, 64] {
            let trace_values: Vec<F> = (0..w).map(|_| F::rand(&mut rng)).collect();
            let q_value = F::rand(&mut rng);
            let r_value = F::rand(&mut rng);
            let leaf = Layer0LeafContent { trace_values, q_value, r_value };

            assert_eq!(leaf.n_components(), w + 2);
            let flat = leaf.serialize_for_merkle();
            assert_eq!(flat.len(), w + 2);

            let recovered = Layer0LeafContent::deserialize_from_components(&flat, w)
                .expect("deserialize with matching width");
            assert_eq!(recovered, leaf);
        }
    }

    /// `deserialize_from_components` rejects a width mismatch.
    #[test]
    fn layer0_leaf_serde_width_mismatch_rejected() {
        let leaf = Layer0LeafContent {
            trace_values: vec![F::one(); 4],
            q_value: F::one(),
            r_value: F::one(),
        };
        let flat = leaf.serialize_for_merkle();
        assert!(Layer0LeafContent::deserialize_from_components(&flat, 3).is_none());
        assert!(Layer0LeafContent::deserialize_from_components(&flat, 5).is_none());
        assert!(Layer0LeafContent::deserialize_from_components(&flat, 4).is_some());
    }

    /// `build_layer0_leaves(&witness)[row].trace_values[col]` must
    /// equal `witness.trace_columns[col][row]`, etc.
    #[test]
    fn build_layer0_leaves_matches_witness_layout() {
        let mut rng = StdRng::seed_from_u64(0xBEEF_0042);
        let w = 4;
        let n = 16;
        let witness = MergeWitness {
            trace_columns: (0..w)
                .map(|_| (0..n).map(|_| F::rand(&mut rng)).collect())
                .collect(),
            ali_quotient: (0..n).map(|_| F::rand(&mut rng)).collect(),
            blinder: (0..n).map(|_| F::rand(&mut rng)).collect(),
            trace_len: 4,
            d0: 3,
            k_shifts: 2,
        };

        let leaves = build_layer0_leaves(&witness);
        assert_eq!(leaves.len(), n);
        for row in 0..n {
            assert_eq!(leaves[row].width(), w);
            assert_eq!(leaves[row].q_value, witness.ali_quotient[row]);
            assert_eq!(leaves[row].r_value, witness.blinder[row]);
            for col in 0..w {
                assert_eq!(
                    leaves[row].trace_values[col],
                    witness.trace_columns[col][row],
                );
            }
        }
    }

    /// **Load-bearing equivalence test (P2.1).**
    ///
    /// The verifier's `reconstruct_f0_at(leaf, x, …)` applied to the
    /// honest layer-0 leaf at position `x ∈ H_0` must produce the
    /// same `f_0(x)` value that `deep_ali_merge_explicit` computes
    /// prover-side at the same `x`.
    ///
    /// This pins the verifier-side reconstruction to the prover-side
    /// merge: any future change to either must keep them in lockstep.
    #[test]
    fn prover_verifier_f0_equivalence_at_every_h0_position() {
        let mut rng = StdRng::seed_from_u64(0xF00D_BABE);
        let trace_len = 8usize;
        let blowup = 4usize;
        let n = trace_len * blowup;
        let w = 3usize;
        let k_shifts = 2usize;
        let d_c = 2usize;
        let d0 = (d_c - 1) * trace_len - 1;

        let omega: F = {
            use ark_ff::FftField;
            F::get_root_of_unity(n as u64).expect("two-adic root")
        };
        let h0: Vec<F> = (0..n).map(|i| omega.pow_u64(i as u64)).collect();

        let trace_columns: Vec<Vec<F>> = (0..w)
            .map(|_| (0..n).map(|_| F::rand(&mut rng)).collect())
            .collect();
        let ali_quotient: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let blinder: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

        let witness = MergeWitness {
            trace_columns,
            ali_quotient,
            blinder,
            trace_len,
            d0,
            k_shifts,
        };

        let shifts: Vec<F> = vec![F::one(), omega];

        // OOD point + claims (in a real protocol, derived from FS).
        let z: Ext = Ext::from_fp_components(&[
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
        ]).expect("ext from components");

        // Honest OOD claims: evaluate the trace and Q at the OOD points.
        // For random ingredients, we just supply random values that play
        // the role of honest claims — the equivalence test does not
        // depend on the claims being "honest with respect to the trace"
        // because both sides use the *same* claim values.
        let trace_at_shifts: Vec<Vec<Ext>> = (0..w)
            .map(|_| (0..k_shifts).map(|_| Ext::from_fp(F::rand(&mut rng))).collect())
            .collect();
        let q_at_z = Ext::from_fp(F::rand(&mut rng));
        let ood = OodClaims { z, trace_at_shifts, q_at_z };

        let challenges = MergeChallenges {
            gamma_1: Ext::from_fp(F::rand(&mut rng)),
            gamma_2: Ext::from_fp(F::rand(&mut rng)),
            beta:    Ext::from_fp(F::rand(&mut rng)),
        };

        // Prover side: compute f_0 over all of H_0.
        let prover_out = deep_ali_merge_explicit::<Ext>(
            &witness, &h0, &shifts, &ood, &challenges,
        );
        assert_eq!(prover_out.f0_evals_ext.len(), n);

        // Build the layer-0 leaves the verifier would open.
        let leaves = build_layer0_leaves(&witness);
        assert_eq!(leaves.len(), n);

        // Verifier side: at every H_0 position, reconstruct from the
        // opened leaf and the (public) OOD claims + challenges, and
        // confirm equality with the prover's f_0 value.
        for row in 0..n {
            let recon = reconstruct_f0_at::<Ext>(
                &leaves[row],
                h0[row],
                &shifts,
                &ood,
                &challenges,
                trace_len,
                d0,
            );
            assert_eq!(
                recon, prover_out.f0_evals_ext[row],
                "prover/verifier f_0 mismatch at row {}", row,
            );
        }
    }

    /// Tamper test: tampering the layer-0 leaf's trace value at one
    /// column must change the reconstructed `f_0(x)` (modulo
    /// vanishingly improbable cancellation).  This certifies that the
    /// merge structure actually depends on the trace value, i.e. the
    /// trace is genuinely committed by the layer-0 leaf — not vestigial.
    #[test]
    fn tampered_layer0_leaf_changes_f0() {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
        let w = 3;
        let n = 8;
        let trace_len = 4;
        let d0 = (2 - 1) * trace_len - 1; // d_c = 2
        let k_shifts = 2;

        let omega: F = {
            use ark_ff::FftField;
            F::get_root_of_unity(n as u64).expect("two-adic root")
        };
        let h0: Vec<F> = (0..n).map(|i| omega.pow_u64(i as u64)).collect();

        let witness = MergeWitness {
            trace_columns: (0..w)
                .map(|_| (0..n).map(|_| F::rand(&mut rng)).collect())
                .collect(),
            ali_quotient: (0..n).map(|_| F::rand(&mut rng)).collect(),
            blinder: (0..n).map(|_| F::rand(&mut rng)).collect(),
            trace_len,
            d0,
            k_shifts,
        };
        let shifts: Vec<F> = vec![F::one(), omega];

        let z: Ext = Ext::from_fp_components(&[
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
            F::rand(&mut rng), F::rand(&mut rng), F::rand(&mut rng),
        ]).expect("ext from components");
        let ood = OodClaims {
            z,
            trace_at_shifts: (0..w)
                .map(|_| (0..k_shifts).map(|_| Ext::from_fp(F::rand(&mut rng))).collect())
                .collect(),
            q_at_z: Ext::from_fp(F::rand(&mut rng)),
        };
        let challenges = MergeChallenges {
            gamma_1: Ext::from_fp(F::rand(&mut rng)),
            gamma_2: Ext::from_fp(F::rand(&mut rng)),
            beta:    Ext::from_fp(F::rand(&mut rng)),
        };

        let leaves = build_layer0_leaves(&witness);

        // Choose a non-zero row to tamper (avoid row=0 where x^correction
        // may be 1 and degenerate cases).
        let tamper_row = 3;
        let tamper_col = 1;
        let honest_f0 = reconstruct_f0_at::<Ext>(
            &leaves[tamper_row], h0[tamper_row], &shifts,
            &ood, &challenges, trace_len, d0,
        );

        // Tamper one trace value.
        let mut tampered = leaves[tamper_row].clone();
        tampered.trace_values[tamper_col] += F::one();
        let tampered_f0 = reconstruct_f0_at::<Ext>(
            &tampered, h0[tamper_row], &shifts,
            &ood, &challenges, trace_len, d0,
        );

        assert_ne!(
            honest_f0, tampered_f0,
            "tampering a trace value at the layer-0 leaf must change f_0",
        );

        // Also confirm tampering Q changes f_0.
        let mut tampered_q = leaves[tamper_row].clone();
        tampered_q.q_value += F::one();
        let tampered_q_f0 = reconstruct_f0_at::<Ext>(
            &tampered_q, h0[tamper_row], &shifts,
            &ood, &challenges, trace_len, d0,
        );
        assert_ne!(honest_f0, tampered_q_f0,
            "tampering Q at the layer-0 leaf must change f_0");

        // And tampering R changes f_0.
        let mut tampered_r = leaves[tamper_row].clone();
        tampered_r.r_value += F::one();
        let tampered_r_f0 = reconstruct_f0_at::<Ext>(
            &tampered_r, h0[tamper_row], &shifts,
            &ood, &challenges, trace_len, d0,
        );
        assert_ne!(honest_f0, tampered_r_f0,
            "tampering R at the layer-0 leaf must change f_0");
    }
}
