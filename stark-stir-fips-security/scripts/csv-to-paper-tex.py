#!/usr/bin/env python3
"""
csv-to-paper-tex.py — emit LaTeX `tabular` fragments matching the
STIR paper's Tables 5 (crypto AIRs), 7 (HashRollup), and 8
(recursive rollup) directly from the bench CSVs.

Inputs (relative to repo root):
  scripts/results/signatures_table.csv
      (produced by scripts/bench-all-signatures.sh)
  scripts/results/ml-dsa-rollup-bench.csv
      (produced by scripts/bench-ml-dsa-rollup.sh)
  scripts/results/ml-dsa-recursive-rollup.csv   [optional]
      (produced by scripts/bench-ml-dsa-recursive-rollup.sh; if absent,
       the recursive-rollup fragment is skipped)

Outputs (under scripts/results/paper_tex/):
  paper-table-crypto-airs.tex      (Table 5 body in paper)
  paper-table-rollup.tex           (Table 7 body)
  paper-table-rollup-recursive.tex (Table 8 body, if input present)

The output is the tabular *body* (\\toprule ... \\bottomrule, NO
\\begin{table}/\\caption/\\label) so the paper source includes them
via `\\input{paper-table-crypto-airs.tex}` inside the existing table
environments — captions, labels, and column-width tweaks stay in the
paper.

Usage:
  python3 scripts/csv-to-paper-tex.py
  python3 scripts/csv-to-paper-tex.py --results-dir custom/dir
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# ─── Output formatting ────────────────────────────────────────────────


def fmt_kib(kib: float) -> str:
    """Render a KiB value. Tables 5+7 use KiB; ML-DSA cells exceed 1 MiB."""
    if kib >= 1024.0:
        return f"{kib / 1024.0:.1f}\\,MiB"
    return f"{kib:.0f}\\,KiB"


def fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "---"
    if ms >= 1000.0:
        return f"{ms / 1000.0:.2f}"
    if ms >= 10.0:
        return f"{ms:.1f}"
    return f"{ms:.2f}"


def fmt_s(s: float) -> str:
    return f"{s:.1f}"


# ─── Table 5: crypto AIRs (signatures_table.csv) ──────────────────────


# Hash → LaTeX macro
HASH_MACRO = {
    "SHA3-256": "\\SHAt",
    "SHA3-384": "\\SHAf",
    "SHA3-512": "\\SHAs",
    "sha3-256": "\\SHAt",
    "sha3-384": "\\SHAf",
    "sha3-512": "\\SHAs",
}


def crypto_airs_table(csv_path: Path) -> str:
    """
    Build the body of Table 5 (tab:crypto-airs) from
    signatures_table.csv.  Schema:
      scheme,nist_level,hash,ext_field,ldt,prove_ms,verify_ms,proof_kib,note
    """
    if not csv_path.is_file():
        return f"% {csv_path} not found — Table 5 fragment skipped\n"

    # Aggregate by (scheme, ldt) — take median of repeat runs if any.
    cells: dict[tuple[str, str], dict] = {}
    nist_level: dict[str, str] = {}
    hash_for: dict[str, str] = {}
    raw: dict[tuple[str, str], list[tuple]] = defaultdict(list)

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            scheme = row["scheme"].strip()
            ldt = row["ldt"].strip().lower()
            nist_level[scheme] = row["nist_level"].strip()
            hash_for[scheme] = row["hash"].strip()

            def _f(x: str) -> float | None:
                if x.strip() in ("", "NA", "N/A", "nan"):
                    return None
                try:
                    return float(x)
                except ValueError:
                    return None

            raw[(scheme, ldt)].append(
                (
                    _f(row["prove_ms"]),
                    _f(row["verify_ms"]),
                    _f(row["proof_kib"]),
                )
            )

    def _median(values):
        clean = [v for v in values if v is not None]
        if not clean:
            return None
        return statistics.median(clean)

    for key, runs in raw.items():
        prove = _median([r[0] for r in runs])
        verify = _median([r[1] for r in runs])
        proof = _median([r[2] for r in runs])
        cells[key] = {
            "prove_ms": prove,
            "verify_ms": verify,
            "proof_kib": proof,
        }

    # Schemes in paper order.  Add new schemes here as the bench grows.
    scheme_order = [
        "RSA-2048",
        "Ed25519",
        "ECDSA-p256",
        "ML-DSA-44",
        "ML-DSA-65",
        "ML-DSA-87",
    ]
    paper_air_name = {
        "RSA-2048": "RSA-2048",
        "Ed25519": "Ed25519",
        "ECDSA-p256": "ECDSA-p256",
        "ML-DSA-44": "ML-DSA-44 v2",
        "ML-DSA-65": "ML-DSA-65 v2",
        "ML-DSA-87": "ML-DSA-87 v2",
    }

    lines = []
    lines.append("\\toprule")
    lines.append(
        "AIR & Lvl & Hash & LDT & Proof & $t_v$ (ms) & $t_p$ (s) \\\\"
    )
    lines.append("\\midrule")

    first = True
    for scheme in scheme_order:
        # Skip schemes that have no rows in the CSV (graceful).
        if not any((scheme, l) in cells for l in ("stir", "fri")):
            continue
        if not first:
            lines.append("\\addlinespace")
        first = False
        for ldt in ("stir", "fri"):
            c = cells.get((scheme, ldt))
            if c is None:
                lines.append(
                    f"% missing: {scheme} / {ldt} not in {csv_path.name}"
                )
                continue
            tp_s = c["prove_ms"] / 1000.0 if c["prove_ms"] is not None else None
            lines.append(
                "  {air:<14}& {lvl} & {h} & {ldt} & {proof} & {tv} & {tp} \\\\".format(
                    air=paper_air_name[scheme],
                    lvl=nist_level[scheme],
                    h=HASH_MACRO.get(hash_for[scheme], hash_for[scheme]),
                    ldt=ldt.upper(),
                    proof=fmt_kib(c["proof_kib"]) if c["proof_kib"] else "---",
                    tv=fmt_ms(c["verify_ms"]),
                    tp=fmt_s(tp_s) if tp_s is not None else "---",
                )
            )
    lines.append("\\bottomrule")
    return "\n".join(lines) + "\n"


# ─── Table 7: HashRollup (ml-dsa-rollup-bench.csv) ────────────────────


def rollup_table(csv_path: Path) -> str:
    """
    Build the body of Table 7 (tab:rollup) from ml-dsa-rollup-bench.csv.
    Schema (the columns we use):
      n,blowup,outer_ldt,outer_prove_ms,outer_verify_ms,outer_size_kib,
      outer_overhead_pct_size
    """
    if not csv_path.is_file():
        return f"% {csv_path} not found — Table 7 fragment skipped\n"

    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "n": int(r["n"]),
                    "ldt": r["outer_ldt"].strip().lower(),
                    "prove_ms": float(r["outer_prove_ms"]),
                    "verify_ms": float(r["outer_verify_ms"]),
                    "size_kib": float(r["outer_size_kib"]),
                    "overhead": float(r["outer_overhead_pct_size"]),
                }
            )

    # Group by N, emit fri then stir (paper order).
    by_n: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_n[r["n"]][r["ldt"]] = r

    lines = []
    lines.append("\\toprule")
    lines.append(
        "$N$ & Outer LDT & Outer Prove (ms) & Outer Verify (ms) & Outer (KiB) & Overhead \\\\"
    )
    lines.append("\\midrule")

    for n in sorted(by_n):
        for ldt in ("fri", "stir"):
            r = by_n[n].get(ldt)
            if r is None:
                continue
            lines.append(
                "  {n} & {ldt} & {tp} & {tv} & {kib} & {oh}\\% \\\\".format(
                    n=n,
                    ldt=ldt.upper(),
                    tp=f"{r['prove_ms']:.1f}",
                    tv=f"{r['verify_ms']:.2f}",
                    kib=f"{r['size_kib']:.1f}",
                    oh=f"{r['overhead']:.2f}",
                )
            )
    lines.append("\\bottomrule")
    return "\n".join(lines) + "\n"


# ─── Table 8: recursive rollup compression ────────────────────────────


def rollup_recursive_table(csv_path: Path) -> str:
    """
    Build the body of Table 8 (tab:rollup-recursive) from
    ml-dsa-recursive-rollup.csv if present.  Expected schema:
      outer_blowup,r,prove_ms,verify_ms,per_sig_kib,bundle_kib,compression_x
    """
    if not csv_path.is_file():
        return f"% {csv_path} not found — Table 8 fragment skipped\n"

    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "bw": int(r["outer_blowup"]),
                    "r": int(r["r"]),
                    "prove_ms": float(r["prove_ms"]),
                    "verify_ms": float(r["verify_ms"]),
                    "per_sig_kib": float(r["per_sig_kib"]),
                    "bundle_kib": float(r["bundle_kib"]),
                    "compression": float(r["compression_x"]),
                }
            )

    rows.sort(key=lambda x: x["bw"])

    lines = []
    lines.append("\\toprule")
    lines.append(
        "Outer bw & $r$ & Prove (ms) & Verify (ms) & per-sig (KiB) & Bundle (KiB) & Compression \\\\"
    )
    lines.append("\\midrule")
    for r in rows:
        lines.append(
            "  {bw:>2} & {r:>3} & {tp} & {tv} & {ps} & {bun} & {cx}$\\times$ \\\\".format(
                bw=r["bw"],
                r=r["r"],
                tp=f"{r['prove_ms']:.0f}",
                tv=f"{r['verify_ms']:.2f}",
                ps=f"{r['per_sig_kib']:.0f}",
                bun=f"{r['bundle_kib']:.0f}",
                cx=f"{r['compression']:.1f}",
            )
        )
    lines.append("\\bottomrule")
    return "\n".join(lines) + "\n"


# ─── Driver ───────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        default="scripts/results",
        help="Directory holding the bench CSVs (default: scripts/results)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Where to write .tex fragments "
            "(default: <results-dir>/paper_tex)"
        ),
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else results_dir / "paper_tex"
    out_dir.mkdir(parents=True, exist_ok=True)

    sig_csv = results_dir / "signatures_table.csv"
    rollup_csv = results_dir / "ml-dsa-rollup-bench.csv"
    recursive_csv = results_dir / "ml-dsa-recursive-rollup.csv"

    artifacts = [
        ("paper-table-crypto-airs.tex", crypto_airs_table(sig_csv)),
        ("paper-table-rollup.tex", rollup_table(rollup_csv)),
        ("paper-table-rollup-recursive.tex", rollup_recursive_table(recursive_csv)),
    ]

    for name, body in artifacts:
        out_path = out_dir / name
        header = (
            f"% Generated by scripts/csv-to-paper-tex.py — DO NOT EDIT.\n"
            f"% Source CSV: {results_dir}\n"
        )
        out_path.write_text(header + body)
        print(f"[wrote] {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
