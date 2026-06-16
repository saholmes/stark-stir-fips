# AWS c5.4xlarge bench launch runbook

Purpose: reproduce the six Table 5 cells of the STIR paper on the
hardware the paper claims (AWS c5.4xlarge, Cascade Lake, AVX-512).
Local laptop runs already validated the k=4 STIR pipeline end-to-end;
this runbook executes the same six bench commands on a server so the
absolute numbers in the published Table 5 match the disclosed hardware.

Total wall-clock: ~80-100 minutes. Total instance cost at ~$0.68/hr:
~$1.20 (round up to $2 if compile cache is cold).

## Files in this package

| File | What it is | Where it goes on AWS |
|---|---|---|
| `end_to_end.rs` | Patched bench harness with 4x4x4 STIR preset + env-var-driven r and output filename | `crates/channel/benches/end_to_end.rs` |
| `channel-Cargo.toml` | channel/Cargo.toml with the `ext-octic` feature added | `crates/channel/Cargo.toml` |
| `run-k4-sweep.sh` | Sequential 6-cell run script | Repository root |
| `k4-bench.patch` | The same changes as a unified diff (alternative to file replacement) | Repository root |

## Prerequisites on the AWS box

```bash
# Ubuntu 22.04 LTS recommended. From a fresh c5.4xlarge:
sudo apt-get update
sudo apt-get install -y build-essential git curl pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

## Launch (assuming the repo lives at `~/stark-stir-fips/stark-stir-fips-security`)

```bash
# 1. Clone the codebase
cd ~
git clone https://github.com/saholmes/stark-nist-fips.git stark-stir-fips
cd stark-stir-fips/stark-stir-fips-security

# 2. Apply the patch (option A: file replacement)
scp $LOCAL:aws-bench-package/end_to_end.rs   crates/channel/benches/end_to_end.rs
scp $LOCAL:aws-bench-package/channel-Cargo.toml crates/channel/Cargo.toml
scp $LOCAL:aws-bench-package/run-k4-sweep.sh ./run-k4-sweep.sh
chmod +x run-k4-sweep.sh

# OR option B: apply as unified diff
# git apply k4-bench.patch

# 3. Sanity check (~30s build + ~5s run)
BENCH_R=54 BENCH_CSV_OUT=/tmp/smoke.csv \
  cargo bench --bench end_to_end -p channel \
    --no-default-features --features parallel,sha3-256 -- --quick
head /tmp/smoke.csv

# 4. Launch the full sweep in the background
nohup ./run-k4-sweep.sh > sweep-stdout.log 2>&1 &
echo "Sweep PID: $!"

# 5. Monitor progress
tail -f /tmp/k4-bench/sweep.log
# OR (the script writes to /Volumes/SAHexternal/Downloads/k4-bench/sweep.log
#     by default — edit the OUTDIR= line at top of run-k4-sweep.sh to point
#     to a Linux-friendly path like /tmp/k4-bench/ before running on AWS)
```

## EXPECTED OUTPUT — six CSV files

After the sweep finishes (~80-100 minutes), the output dir contains:

```
Final-NIST-L1-k4arity-SHA256-FP6-r54-q2-40-benchmarkdata.csv
Final-NIST-L1-k4arity-SHA3384-FP6-r54-q2-65-benchmarkdata.csv
Final-NIST-L1-k4arity-SHA3512-FP6-r54-q2-90-benchmarkdata.csv
Final-NIST-L3-k4arity-SHA384-FP6-r79-q2-40-benchmarkdata.csv
Final-NIST-L3-k4arity-SHA512-FP6-r79-q2-90-q2-65-benchmarkdata.csv
Final-NIST-L5-k4arity-SHA512-FP8-r105-multiair-benchmarkdata.csv
sweep.log
```

Each CSV contains 20 rows: 10 binary FRI (`2power16`, k=16..25) + 10 STIR
k=4 (`4x4x4`, k=16..25). For the paper's Table 5, the canonical row is
**4x4x4 at k=22** (LDE = 2^22 = T·d_c/ρ_0 = 2^16 · 2 · 32).

## Transferring CSVs back to local

```bash
# From your local machine, pull the CSVs back:
scp -r aws-user@$AWS_IP:~/stark-stir-fips/stark-stir-fips-security/k4-bench-aws/ \
  /Volumes/SAHexternal/Downloads/k4-bench-aws/
```

## Side-by-side reference: laptop numbers (Apple Silicon)

The local-laptop run gave these L1 SHA-3-256 figures at k=22 for
sanity-checking the AWS run won't shock you:

| Cell | Proof (KiB) | Verify (ms) | Prove (s) |
|---|---|---|---|
| L1 SHA-3-256 (`4x4x4`, k=22) | 193 | 0.79 | 6.99 |

AWS c5.4xlarge with AVX-512 Keccak-tiny will likely produce:
- Proof size: **identical to laptop** (deterministic, no platform dependence)
- Verify time: **0.5-1.5 ms** range (faster Keccak via AVX-512)
- Prove time: **5-15 s** range (FFT FFI competitive between Apple Silicon
  performance cores and Cascade Lake)

If proof sizes differ from the laptop, that's a bug — should be byte-identical.
If verify is more than ~3× slower than laptop, AVX-512 path is not being
exercised (check `cat /proc/cpuinfo | grep avx512`).

## Paper update path once CSVs arrive

When you hand me the six AWS CSVs, I'll:

1. Extract each `4x4x4` k=22 row
2. Replace the six rows in Table 5 (main-24.tex lines 1089-1096)
3. Confirm/update the abstract / §7.1 / §8.1 / §9 if the numbers shift
   beyond ±10%
4. The Table 5 caption already correctly says "AWS c5.4xlarge, 16 vCPU,
   AVX-512, 16 Rayon threads" — that becomes accurate once the AWS data
   replaces the laptop data

## Risk and rollback

- If a cell trips an assertion mid-sweep, the script's `set -e` halts;
  earlier CSVs are still usable
- All builds are non-default-features so no risk to release builds
- The 4x4x4 preset was previously commented out (line 170 of the unpatched
  end_to_end.rs); we only enabled it, not modified the LDT logic itself
- Worst case: if AWS data shows the k=4 advantage is smaller than laptop
  (e.g., 2× instead of 4-5×), update the ratios in the paper accordingly;
  the soundness analysis is unchanged
