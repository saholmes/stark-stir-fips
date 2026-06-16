//! P2.2 — Layer-0 Merkle tree wrapper for the explicit merge.
//!
//! Wraps the existing `merkle::MerkleTreeChannel` so the prover can
//! commit to the layer-0 leaves `(T_1(x), …, T_w(x), Q(x), R(x))`
//! produced by `explicit_merge::build_layer0_leaves`, and so the
//! verifier can open a single position, check the Merkle path, and
//! reconstruct `f_0(x)` via `explicit_merge::reconstruct_f0_at`.
//!
//! ## Why a separate file
//!
//! `explicit_merge.rs` is the *construction-layer* code (witness,
//! OOD claims, FS challenges, eq:merge evaluation).  This file is the
//! *commitment-layer* code (Merkle tree build, root, open, verify).
//! Keeping them separate makes the dependency on the `merkle` crate
//! local to commitment-layer code, and keeps the construction-layer
//! testable in isolation.
//!
//! ## Scope of this commit
//!
//! Adds the data structures and primitives.  The next commit
//! (P3) wires these into a STIR/FRI prover entry point with the
//! verifier-side ALI consistency check (eq:ali-check).

use ark_goldilocks::Goldilocks as F;
use hash::HASH_BYTES;
use merkle::{MerkleChannelCfg, MerkleOpening, MerkleTreeChannel};

use crate::explicit_merge::{
    build_layer0_leaves, reconstruct_f0_at, Layer0LeafContent, MergeChallenges,
    MergeWitness, OodClaims,
};
use crate::tower_field::TowerField;

// ─── Layer-0 commit ──────────────────────────────────────────────

/// A built Merkle tree over the layer-0 leaves.
///
/// Carries:
///   - the Merkle tree itself (so the prover can `open` later),
///   - the root,
///   - the original `Layer0LeafContent` per position (needed when the
///     verifier asks for an opening — we recover the leaf payload
///     from the index).
pub struct Layer0Commit {
    /// Internal Merkle tree.  Built with binary arity throughout.
    pub tree: MerkleTreeChannel,

    /// Final Merkle root.
    pub root: [u8; HASH_BYTES],

    /// Per-position leaf contents.  Length `|H_0|`.
    ///
    /// Stored so the prover can produce an opening on demand without
    /// re-running `build_layer0_leaves`.
    pub leaves: Vec<Layer0LeafContent>,

    /// AIR trace width `w`.  Stored so the verifier can confirm the
    /// opening's serialised payload deserialises to the expected
    /// width.
    pub trace_width: usize,
}

impl Layer0Commit {
    /// Build the layer-0 Merkle tree from a `MergeWitness`.
    ///
    /// `tree_label` is a domain-separation tag for the Merkle tree
    /// (use a different value per layer / per proof to prevent
    /// cross-tree replay attacks).
    pub fn from_witness(witness: &MergeWitness, tree_label: u64) -> Self {
        let leaves = build_layer0_leaves(witness);
        Self::from_leaves(&leaves, tree_label)
    }

    /// Build the layer-0 Merkle tree from pre-built `Layer0LeafContent`
    /// leaves.  Useful for tests; the `from_witness` constructor is
    /// the production entry point.
    pub fn from_leaves(leaves: &[Layer0LeafContent], tree_label: u64) -> Self {
        assert!(!leaves.is_empty(), "Layer0Commit: cannot build from zero leaves");

        let width = leaves[0].width();
        for (i, l) in leaves.iter().enumerate() {
            debug_assert_eq!(
                l.width(), width,
                "Layer0Commit: leaf {} has width {}, expected {}",
                i, l.width(), width,
            );
        }

        // Serialise every leaf into Vec<F> for the Merkle hasher.
        let leaf_payloads: Vec<Vec<F>> =
            leaves.iter().map(|l| l.serialize_for_merkle()).collect();

        // Build the tree with binary arity at every level.  Matches the
        // existing convention in stir_halve.rs.
        let n = leaf_payloads.len();
        let depth = (n.next_power_of_two().trailing_zeros() as usize).max(1);
        let arities: Vec<usize> = std::iter::repeat(2).take(depth).collect();
        let cfg = MerkleChannelCfg::new(arities, tree_label);

        let mut tree = MerkleTreeChannel::new(cfg, [0u8; HASH_BYTES]);
        tree.push_leaves_parallel(&leaf_payloads);
        let root = tree.finalize();

        Self {
            tree,
            root,
            leaves: leaves.to_vec(),
            trace_width: width,
        }
    }

    /// Open the layer-0 leaf at position `index`.
    ///
    /// Returns the leaf contents together with the Merkle authentication
    /// path.  The verifier reconstructs `f_0(index_x)` from the leaf via
    /// `Layer0Opening::reconstruct_f0`.
    pub fn open(&self, index: usize) -> Layer0Opening {
        assert!(
            index < self.leaves.len(),
            "Layer0Commit::open: index {} >= |H_0| = {}",
            index, self.leaves.len(),
        );
        let merkle_opening = self.tree.open(index);
        Layer0Opening {
            index,
            leaf: self.leaves[index].clone(),
            merkle_opening,
            trace_width: self.trace_width,
        }
    }
}

// ─── Layer-0 opening (proof + leaf) ──────────────────────────────

/// A layer-0 opening at one queried position.  Sent over the wire from
/// prover to verifier.
#[derive(Debug, Clone)]
pub struct Layer0Opening {
    /// Position index in `H_0`.
    pub index: usize,

    /// The opened layer-0 leaf payload `(T_1, …, T_w, Q, R)`.
    pub leaf: Layer0LeafContent,

    /// Merkle authentication path produced by `MerkleTreeChannel::open`.
    pub merkle_opening: MerkleOpening,

    /// AIR trace width `w`, captured at build time so the verifier can
    /// cross-check the deserialised payload shape.
    pub trace_width: usize,
}

impl Layer0Opening {
    /// Verify the Merkle authentication path against the published
    /// root.  Does *not* reconstruct `f_0`; that is a separate step
    /// the caller invokes once the Merkle proof has verified.
    pub fn verify_merkle(
        &self,
        cfg: &MerkleChannelCfg,
        root: [u8; HASH_BYTES],
    ) -> bool {
        // The leaf hash inside MerkleOpening is what the prover sent;
        // re-derive it from the *claimed* leaf payload and check that
        // the prover did not tamper with it relative to the Merkle path
        // they sent.
        let claimed_payload = self.leaf.serialize_for_merkle();
        let recomputed_leaf_hash =
            merkle::compute_leaf_hash(cfg, self.index, &claimed_payload);
        if recomputed_leaf_hash != self.merkle_opening.leaf {
            return false;
        }

        MerkleTreeChannel::verify_opening(cfg, root, &self.merkle_opening, &[0u8; HASH_BYTES])
    }

    /// Verify the Merkle path AND reconstruct `f_0(x)` from the opened
    /// leaf.  Returns `Some(f_0(x))` on success, `None` on Merkle
    /// failure.
    ///
    /// The caller's outer FRI / STIR verifier compares the returned
    /// `f_0(x)` against the next-round fold.
    pub fn verify_and_reconstruct<E: TowerField>(
        &self,
        cfg: &MerkleChannelCfg,
        root: [u8; HASH_BYTES],
        x: F,
        shifts: &[F],
        ood: &OodClaims<E>,
        ch: &MergeChallenges<E>,
        trace_len: usize,
        d0: usize,
    ) -> Option<E> {
        if !self.verify_merkle(cfg, root) {
            return None;
        }
        if self.leaf.width() != self.trace_width {
            return None;
        }
        Some(reconstruct_f0_at::<E>(
            &self.leaf, x, shifts, ood, ch, trace_len, d0,
        ))
    }
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit_merge::deep_ali_merge_explicit;
    use crate::sextic_ext::SexticExt;
    use ark_ff::{FftField, One, UniformRand};
    use rand::{rngs::StdRng, SeedableRng};

    type Ext = SexticExt;

    /// Build a Merkle tree from a random witness, open every position,
    /// verify the Merkle proof, and confirm the reconstructed `f_0(x)`
    /// matches what `deep_ali_merge_explicit` computes prover-side.
    ///
    /// This is the end-to-end round-trip pin for P2.2.
    #[test]
    fn layer0_commit_open_verify_roundtrip_at_every_position() {
        let mut rng = StdRng::seed_from_u64(0xF1F2_F3F4);
        let trace_len: usize = 8;
        let blowup: usize = 4;
        let n: usize = trace_len * blowup;
        let w = 3usize;
        let k_shifts = 2usize;
        let d_c = 2usize;
        let d0 = (d_c - 1) * trace_len - 1;

        let omega: F = F::get_root_of_unity(n as u64).expect("two-adic root");
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
        let ch = MergeChallenges {
            gamma_1: Ext::from_fp(F::rand(&mut rng)),
            gamma_2: Ext::from_fp(F::rand(&mut rng)),
            beta:    Ext::from_fp(F::rand(&mut rng)),
        };

        // Prover-side: compute f_0 over all of H_0, build the layer-0
        // commit, get its root.
        let prover_f0 = deep_ali_merge_explicit::<Ext>(
            &witness, &h0, &shifts, &ood, &ch,
        );
        let layer0_tree_label = 0xABCD_0001u64;
        let commit = Layer0Commit::from_witness(&witness, layer0_tree_label);
        let root = commit.root;

        // The Merkle config the verifier will use.
        let depth = (n.next_power_of_two().trailing_zeros() as usize).max(1);
        let cfg = MerkleChannelCfg::new(vec![2usize; depth], layer0_tree_label);

        // Verifier-side: at every position, open the leaf, verify the
        // Merkle path, and reconstruct f_0.  Must equal prover's f_0.
        for index in 0..n {
            let opening = commit.open(index);

            let recon = opening
                .verify_and_reconstruct::<Ext>(
                    &cfg, root, h0[index],
                    &shifts, &ood, &ch, trace_len, d0,
                )
                .expect("Merkle verification + reconstruction succeeded");

            assert_eq!(
                recon, prover_f0.f0_evals_ext[index],
                "reconstructed f_0 mismatch at index {} (H_0 position {:?})",
                index, h0[index],
            );
        }
    }

    /// Tamper test: flipping a leaf payload after opening must cause
    /// `verify_merkle` (and thus `verify_and_reconstruct`) to reject.
    #[test]
    fn tampered_leaf_payload_fails_merkle_verify() {
        let mut rng = StdRng::seed_from_u64(0xBADD_BADD);
        let n: usize = 16;
        let w: usize = 4;
        let leaves: Vec<Layer0LeafContent> = (0..n).map(|_| Layer0LeafContent {
            trace_values: (0..w).map(|_| F::rand(&mut rng)).collect(),
            q_value: F::rand(&mut rng),
            r_value: F::rand(&mut rng),
        }).collect();

        let label = 0x1234_5678u64;
        let commit = Layer0Commit::from_leaves(&leaves, label);
        let depth = (n.next_power_of_two().trailing_zeros() as usize).max(1);
        let cfg = MerkleChannelCfg::new(vec![2usize; depth], label);

        // Honest opening verifies.
        let honest = commit.open(7);
        assert!(honest.verify_merkle(&cfg, commit.root));

        // Tamper the leaf's trace value — same Merkle path, but the
        // leaf hash will mismatch.
        let mut tampered = honest.clone();
        tampered.leaf.trace_values[0] += F::one();
        assert!(!tampered.verify_merkle(&cfg, commit.root));

        // Tamper Q.
        let mut tampered_q = honest.clone();
        tampered_q.leaf.q_value += F::one();
        assert!(!tampered_q.verify_merkle(&cfg, commit.root));

        // Tamper R.
        let mut tampered_r = honest.clone();
        tampered_r.leaf.r_value += F::one();
        assert!(!tampered_r.verify_merkle(&cfg, commit.root));
    }

    /// Width-mismatch test: an opening from a tree of width `w` must
    /// fail verification when the verifier expects a different width
    /// (caught at the deserialisation step).
    #[test]
    fn width_mismatch_in_opening_rejected() {
        let mut rng = StdRng::seed_from_u64(0xCAFEBABE);
        let n: usize = 8;
        let w: usize = 4;
        let leaves: Vec<Layer0LeafContent> = (0..n).map(|_| Layer0LeafContent {
            trace_values: (0..w).map(|_| F::rand(&mut rng)).collect(),
            q_value: F::rand(&mut rng),
            r_value: F::rand(&mut rng),
        }).collect();
        let label = 0xDEAD_DEADu64;
        let commit = Layer0Commit::from_leaves(&leaves, label);
        let depth = (n.next_power_of_two().trailing_zeros() as usize).max(1);
        let cfg = MerkleChannelCfg::new(vec![2usize; depth], label);

        let mut opening = commit.open(3);
        assert_eq!(opening.trace_width, w);

        // Verifier supplies the wrong expected width via
        // verify_and_reconstruct (we simulate by mutating the field).
        opening.trace_width = w + 1;

        // Honest Merkle path verifies, but reconstruction is gated by
        // the width check and returns None.
        let z = Ext::from_fp(F::rand(&mut rng));
        let ood = OodClaims {
            z,
            trace_at_shifts: vec![vec![Ext::from_fp(F::rand(&mut rng)); 2]; w + 1],
            q_at_z: Ext::from_fp(F::rand(&mut rng)),
        };
        let ch = MergeChallenges {
            gamma_1: Ext::from_fp(F::rand(&mut rng)),
            gamma_2: Ext::from_fp(F::rand(&mut rng)),
            beta:    Ext::from_fp(F::rand(&mut rng)),
        };
        let result = opening.verify_and_reconstruct::<Ext>(
            &cfg, commit.root, F::one(), &[F::one(), F::one()],
            &ood, &ch, 4, 3,
        );
        assert!(result.is_none(),
            "width mismatch must cause verify_and_reconstruct to return None");
    }

    /// Wrong root rejection: opening verified against a different root
    /// (e.g. a replay from a different proof) must fail.
    #[test]
    fn opening_against_wrong_root_rejected() {
        let mut rng = StdRng::seed_from_u64(0x9999_9999);
        let n: usize = 8;
        let w: usize = 2;
        let leaves_a: Vec<Layer0LeafContent> = (0..n).map(|_| Layer0LeafContent {
            trace_values: (0..w).map(|_| F::rand(&mut rng)).collect(),
            q_value: F::rand(&mut rng),
            r_value: F::rand(&mut rng),
        }).collect();
        let leaves_b: Vec<Layer0LeafContent> = (0..n).map(|_| Layer0LeafContent {
            trace_values: (0..w).map(|_| F::rand(&mut rng)).collect(),
            q_value: F::rand(&mut rng),
            r_value: F::rand(&mut rng),
        }).collect();

        let label = 0x1111_2222u64;
        let commit_a = Layer0Commit::from_leaves(&leaves_a, label);
        let commit_b = Layer0Commit::from_leaves(&leaves_b, label);
        assert_ne!(commit_a.root, commit_b.root);

        let depth = (n.next_power_of_two().trailing_zeros() as usize).max(1);
        let cfg = MerkleChannelCfg::new(vec![2usize; depth], label);
        let opening = commit_a.open(2);

        assert!(opening.verify_merkle(&cfg, commit_a.root));
        assert!(!opening.verify_merkle(&cfg, commit_b.root));
    }
}
