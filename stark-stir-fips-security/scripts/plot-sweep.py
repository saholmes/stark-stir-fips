#!/usr/bin/env python3
"""Plot arity/blowup × NIST-level × hash function sweep results.

Reads scripts/results/sweep.csv and emits paper-ready figures to
scripts/results/figures/.  Figures cover the four headline claims:

  Figure 1: STIR vs FRI at the deployable corner (k=2, max-b)
            per production AIR + NIST level — bar chart.
  Figure 2: NIST-level cost progression at fixed (LDT=STIR, k=2,
            b=max) — line plot proof_kib + verify_ms vs λ.
  Figure 3: (k, b) optimum surfaces per AIR at L1 baseline —
            shows the small-AIR-vs-large-AIR optimum inversion.
  Figure 4: Hash-axis cost progression (L1 q=2^40 → q=2^65 → q=2^90)
            at the deployable corner — line plot per AIR.
  Figure 5: STIR/FRI ratio across all production cells —
            box plot summarising the consistency of STIR's win.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH  = REPO_ROOT / "scripts" / "results" / "sweep.csv"
FIG_DIR   = REPO_ROOT / "scripts" / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Paper-friendly defaults: serif fonts, no top/right spines, tight bbox.
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

# Consistent colour scheme: STIR (blue), FRI (orange).
COL_STIR = "#1f77b4"
COL_FRI  = "#d62728"

# Load + clean.
df = pd.read_csv(CSV_PATH)
df = df[df["prove_ms"] != "NA"].copy()
for col in ("prove_ms", "verify_ms", "proof_kib"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["prove_ms", "verify_ms", "proof_kib"])
df["nist_label"] = df["nist"].map({1: "L1", 3: "L3", 5: "L5"})
print(f"Loaded {len(df)} OK cells; "
      f"AIRs: {sorted(df['air'].unique())}; "
      f"LDTs: {sorted(df['ldt'].unique())}")


# ── Figure 1: STIR vs FRI at deployable corners ──────────────────
def figure_1_deployable_corner():
    """Bar chart of proof_kib and verify_ms at the k=2/max-b corner
    per (AIR, NIST level) for STIR vs FRI.  Baseline q=2^40 only.
    """
    fig, (ax_proof, ax_verify) = plt.subplots(1, 2, figsize=(9, 3.2))

    # Production AIRs and their max b for the corner.
    air_max_b = {"rsa2048": 32, "ed25519": 64, "mldsa_v2": 64}
    air_label = {"rsa2048": "RSA-2048", "ed25519": "Ed25519 K=256", "mldsa_v2": "ML-DSA-v2"}

    # Filter to q=2^40 baseline cells.
    base = df[(df["q_log"] == 40) & (df["k"] == 2)].copy()

    rows = []
    for air, max_b in air_max_b.items():
        for nist in (1, 3, 5):
            for ldt in ("stir", "fri"):
                sub = base[(base["air"] == air) & (base["nist"] == nist)
                           & (base["b"] == max_b) & (base["ldt"] == ldt)]
                if not sub.empty:
                    row = sub.iloc[0]
                    rows.append({"air": air_label[air], "nist": f"L{nist}",
                                 "ldt": ldt, "proof_kib": row.proof_kib,
                                 "verify_ms": row.verify_ms})
    plot_df = pd.DataFrame(rows)

    # Grouped bar chart: per (AIR, NIST), STIR vs FRI.
    air_nist = [(a, n) for a in air_label.values() for n in ("L1", "L3", "L5")]
    x_pos = list(range(len(air_nist)))
    width = 0.35

    for ax, metric, ylabel in [(ax_proof, "proof_kib", "Proof (KiB)"),
                               (ax_verify, "verify_ms", "Verify (ms)")]:
        stir_vals, fri_vals = [], []
        for air, nist in air_nist:
            stir = plot_df[(plot_df.air == air) & (plot_df.nist == nist)
                           & (plot_df.ldt == "stir")][metric]
            fri  = plot_df[(plot_df.air == air) & (plot_df.nist == nist)
                           & (plot_df.ldt == "fri")][metric]
            stir_vals.append(stir.iloc[0] if not stir.empty else 0)
            fri_vals.append(fri.iloc[0] if not fri.empty else 0)

        ax.bar([x - width/2 for x in x_pos], stir_vals,  width, label="STIR",
               color=COL_STIR, edgecolor="black", linewidth=0.4)
        ax.bar([x + width/2 for x in x_pos], fri_vals,   width, label="FRI",
               color=COL_FRI,  edgecolor="black", linewidth=0.4)

        labels = [f"{a.split(' ')[0]}\n{n}" for a, n in air_nist]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", frameon=False)

        # Vertical separators between AIR groups.
        for sep in (2.5, 5.5):
            ax.axvline(sep, color="gray", linewidth=0.4, alpha=0.5)

    ax_proof.set_title("Proof size at k=2, b=max, q=2^40 baseline")
    ax_verify.set_title("Verify time at k=2, b=max, q=2^40 baseline")

    fig.suptitle("Figure 1.  STIR vs FRI at the deployable corner per production AIR",
                 fontsize=10)
    out = FIG_DIR / "fig1_deployable_corner.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


# ── Figure 2: NIST-level cost progression ────────────────────────
def figure_2_nist_progression():
    """Line plot showing proof_kib and verify_ms scaling from
    L1 → L3 → L5 at the deployable corner (STIR, k=2, b=max).
    """
    fig, (ax_proof, ax_verify) = plt.subplots(1, 2, figsize=(9, 3.2))
    air_max_b = {"rsa2048": 32, "ed25519": 64, "mldsa_v2": 64}
    air_style = {"rsa2048": ("o-", "RSA-2048"),
                 "ed25519": ("s-", "Ed25519 K=256"),
                 "mldsa_v2": ("D-", "ML-DSA-v2")}

    base = df[(df["q_log"] == 40) & (df["k"] == 2) & (df["ldt"] == "stir")]
    for air, max_b in air_max_b.items():
        sub = base[(base["air"] == air) & (base["b"] == max_b)
                   ].sort_values("nist")
        if sub.empty:
            continue
        marker, lbl = air_style[air]
        ax_proof.plot(sub["nist_label"], sub["proof_kib"], marker, label=lbl)
        ax_verify.plot(sub["nist_label"], sub["verify_ms"], marker, label=lbl)

    for ax, ylabel, title in [(ax_proof,  "Proof (KiB)", "Proof size"),
                              (ax_verify, "Verify (ms)", "Verify time")]:
        ax.set_xlabel("NIST level (λ ∈ {128, 192, 256})")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", frameon=False, fontsize=8)

    fig.suptitle(
        "Figure 2.  NIST-level cost progression — STIR at k=2, b=max",
        fontsize=10)
    out = FIG_DIR / "fig2_nist_progression.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


# ── Figure 3: (k, b) optimum surfaces per AIR ────────────────────
def figure_3_kb_optimum():
    """Per-AIR heatmaps of proof_kib over (k, b) at L1 baseline,
    showing the small-AIR-vs-large-AIR optimum inversion.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    air_list = [("ecdsa", "ECDSA-K=2 stub (n_trace=8)"),
                ("rsa2048", "RSA-2048 (n_trace=4096)"),
                ("ed25519", "Ed25519 K=256 (n_trace=1024)")]

    base = df[(df["nist"] == 1) & (df["q_log"] == 40)
              & (df["ldt"] == "stir")]

    for ax, (air, title) in zip(axes, air_list):
        sub = base[base["air"] == air]
        if sub.empty:
            ax.set_title(f"{title}\n(no data)"); ax.axis("off"); continue
        # Pivot to (k × b) grid.
        grid = sub.pivot_table(index="k", columns="b",
                               values="proof_kib", aggfunc="mean")
        # Order rows ascending by k, columns ascending by b.
        grid = grid.sort_index(axis=0).sort_index(axis=1)

        im = ax.imshow(grid.values, aspect="auto", cmap="viridis_r",
                       origin="lower")
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels(grid.columns)
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels(grid.index)
        ax.set_xlabel("blowup b")
        ax.set_ylabel("fold arity k")
        ax.set_title(title, fontsize=8)

        # Annotate each cell with the KiB value.
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.0f}",
                            ha="center", va="center",
                            color="white" if v > grid.values.max() / 2 else "black",
                            fontsize=7)
        fig.colorbar(im, ax=ax, label="proof (KiB)",
                     fraction=0.05, pad=0.02)

    fig.suptitle("Figure 3.  Per-AIR (k, b) STIR proof-size surface at "
                 "L1/sha3-256/q=2^40 — optimum inverts between AIR scales",
                 fontsize=10)
    out = FIG_DIR / "fig3_kb_optimum.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


# ── Figure 4: Hash-axis cost progression at L1 ───────────────────
def figure_4_hash_axis():
    """At L1 deployable corner (STIR k=2 b=max), how does increasing
    the hash variant (sha3-256 → sha3-512) and q budget affect cost?
    """
    fig, (ax_proof, ax_verify) = plt.subplots(1, 2, figsize=(9, 3.2))
    air_max_b = {"rsa2048": 32, "ed25519": 64, "mldsa_v2": 64}
    air_style = {"rsa2048": ("o-", "RSA-2048"),
                 "ed25519": ("s-", "Ed25519 K=256"),
                 "mldsa_v2": ("D-", "ML-DSA-v2")}

    # L1 cell-tuples, ordered by q_log.
    base = df[(df["nist"] == 1) & (df["k"] == 2) & (df["ldt"] == "stir")]

    x_labels = ["sha3-256\nq=2^40", "sha3-384\nq=2^65", "sha3-512\nq=2^90"]
    q_order = [40, 65, 90]

    for air, max_b in air_max_b.items():
        marker, lbl = air_style[air]
        proof_vals, verify_vals = [], []
        for q in q_order:
            sub = base[(base["air"] == air) & (base["b"] == max_b)
                       & (base["q_log"] == q)]
            if sub.empty:
                proof_vals.append(None); verify_vals.append(None)
            else:
                proof_vals.append(sub.iloc[0].proof_kib)
                verify_vals.append(sub.iloc[0].verify_ms)
        ax_proof.plot(x_labels, proof_vals, marker, label=lbl)
        ax_verify.plot(x_labels, verify_vals, marker, label=lbl)

    for ax, ylabel, title in [(ax_proof,  "Proof (KiB)", "Proof size"),
                              (ax_verify, "Verify (ms)", "Verify time")]:
        ax.set_xlabel("Binding-stack hash + quantum query budget")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax_proof.tick_params(axis="x", labelsize=7)
    ax_verify.tick_params(axis="x", labelsize=7)

    fig.suptitle(
        "Figure 4.  Hash-axis cost progression at L1 — STIR k=2 b=max, "
        "λ=128 held fixed",
        fontsize=10)
    out = FIG_DIR / "fig4_hash_axis.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


# ── Figure 5: STIR/FRI ratio distribution ────────────────────────
def figure_5_stir_fri_ratio():
    """Box plot of FRI/STIR ratio (proof_kib + verify_ms) across all
    matched cells, demonstrating STIR's consistent win.
    """
    # Pivot to wide form: one row per (nist, q_log, hash, air, k, b)
    # with stir + fri columns.
    keys = ["nist", "q_log", "hash", "air", "k", "b"]
    stir = df[df["ldt"] == "stir"].set_index(keys)
    fri  = df[df["ldt"] == "fri"].set_index(keys)
    joined = stir.join(fri, lsuffix="_stir", rsuffix="_fri", how="inner")
    joined = joined.reset_index()

    joined["proof_ratio"]  = joined["proof_kib_fri"]  / joined["proof_kib_stir"]
    joined["verify_ratio"] = joined["verify_ms_fri"]  / joined["verify_ms_stir"]

    fig, (ax_proof, ax_verify) = plt.subplots(1, 2, figsize=(9, 3.2))
    air_label = {"ecdsa": "ECDSA stub", "rsa2048": "RSA-2048",
                 "ed25519": "Ed25519 K=256", "mldsa_v2": "ML-DSA-v2"}

    for ax, ratio_col, ylabel in [(ax_proof,  "proof_ratio",  "FRI proof / STIR proof"),
                                  (ax_verify, "verify_ratio", "FRI verify / STIR verify")]:
        data, labels = [], []
        for air in ("ecdsa", "rsa2048", "ed25519", "mldsa_v2"):
            sub = joined[joined["air"] == air][ratio_col].dropna()
            if not sub.empty:
                data.append(sub.values)
                labels.append(f"{air_label[air]}\n(n={len(sub)})")
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor=COL_STIR, alpha=0.3),
                        medianprops=dict(color="black", linewidth=1.5))
        ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_ylim(0.5, ax.get_ylim()[1] * 1.2)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.tick_params(axis="x", labelsize=7)

    fig.suptitle(
        "Figure 5.  STIR's proof-size and verify-time wins across all "
        "matched cells (>1 = STIR wins)",
        fontsize=10)
    out = FIG_DIR / "fig5_stir_fri_ratio.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")

    # Headline ratios printed for inclusion in the paper.
    print("  STIR/FRI ratio summary (median across all matched cells):")
    for air in ("ecdsa", "rsa2048", "ed25519", "mldsa_v2"):
        sub = joined[joined["air"] == air]
        if sub.empty:
            continue
        pr = sub["proof_ratio"].median()
        vr = sub["verify_ratio"].median()
        print(f"    {air:10s} n={len(sub):3d}  "
              f"proof ratio median = {pr:.2f}×  verify ratio median = {vr:.2f}×")


if __name__ == "__main__":
    figure_1_deployable_corner()
    figure_2_nist_progression()
    figure_3_kb_optimum()
    figure_4_hash_axis()
    figure_5_stir_fri_ratio()
    print(f"\nAll figures in {FIG_DIR.relative_to(REPO_ROOT)}")
