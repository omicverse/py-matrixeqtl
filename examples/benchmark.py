#!/usr/bin/env python
"""Head-to-head benchmark: pymatrixeqtl vs R MatrixEQTL.

Runs both the Python port and the R package on MatrixEQTL's bundled
example dataset, for all three models, and reports wall-clock time and
accuracy (Pearson r / max |diff| of beta, t/F statistic and p-value).

Usage
-----
    python benchmark.py            # example data, all models
    python benchmark.py --big      # also a larger synthetic dataset (Python only)

The R side is invoked through the CMAP conda env; if R or MatrixEQTL is
not available the script still reports the Python timings.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pymatrixeqtl as me

CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"
R_DRIVER = Path(__file__).parent.parent / "tests" / "r_reference_driver.R"

MODELS = {
    "linear": me.modelLINEAR,
    "anova": me.modelANOVA,
    "cross": me.modelLINEAR_CROSS,
}


def time_python(reps: int = 20):
    """Time the Python engine on the example data (all three models)."""
    ex = me.load_example()
    timings = {}
    results = {}
    for name, model in MODELS.items():
        t0 = time.perf_counter()
        for _ in range(reps):
            res = me.Matrix_eQTL_main(
                snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
                pvOutputThreshold=1.0, pvOutputThreshold_cis=1.0,
                snpspos=ex["snpspos"], genepos=ex["genepos"],
                useModel=model, cisDist=1e6,
            )
        timings[name] = (time.perf_counter() - t0) / reps
        results[name] = {"cis": res.cis, "trans": res.trans}
    return timings, results


def run_r_reference(out_dir: Path):
    """Run the R driver; return (per-model wall time, success flag)."""
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {out_dir}"
    )
    t0 = time.perf_counter()
    res = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    return elapsed, res.returncode == 0


def accuracy(py_results, r_dir: Path):
    """Compare Python vs R result tables; print Pearson r / max |diff|."""
    print(f"\n{'model/table':>16}  {'column':>10}  {'Pearson r':>12}  {'max|diff|':>12}")
    print("-" * 56)
    for model in MODELS:
        for tag in ("cis", "trans"):
            r_path = r_dir / f"{model}_{tag}.tsv"
            if not r_path.exists():
                continue
            r_df = pd.read_csv(r_path, sep="\t")
            py_df = py_results[model][tag]
            m = r_df.merge(py_df, on=["snps", "gene"], suffixes=("_R", "_PY"))
            cols = ["statistic", "pvalue", "FDR"]
            if "beta_R" in m.columns and m["beta_R"].notna().any():
                cols = ["beta"] + cols
            for col in cols:
                r = m[f"{col}_R"].to_numpy()
                p = m[f"{col}_PY"].to_numpy()
                rho = np.corrcoef(r, p)[0, 1]
                md = np.max(np.abs(r - p))
                print(f"{model + '/' + tag:>16}  {col:>10}  "
                      f"{rho:>12.8f}  {md:>12.2e}")


def big_benchmark():
    """Time the Python engine on a larger synthetic dataset."""
    rng = np.random.RandomState(0)
    n_samples, n_snps, n_genes = 200, 5000, 2000
    snps = rng.binomial(2, 0.3, size=(n_snps, n_samples)).astype(float)
    gene = rng.randn(n_genes, n_samples)
    cvrt = rng.randn(3, n_samples)
    snps_sd = me.SlicedData(snps)
    gene_sd = me.SlicedData(gene)
    cvrt_sd = me.SlicedData(cvrt)
    t0 = time.perf_counter()
    res = me.Matrix_eQTL_main(
        snps=snps_sd, gene=gene_sd, cvrt=cvrt_sd,
        pvOutputThreshold=1e-4, useModel=me.modelLINEAR,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nLarge synthetic dataset "
          f"({n_snps} SNPs x {n_genes} genes x {n_samples} samples):")
    print(f"  {n_snps * n_genes:,} pairs tested in {elapsed:.3f} s "
          f"({n_snps * n_genes / elapsed:,.0f} tests/s)")
    print(f"  {len(res.all):,} eQTLs at p < 1e-4")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--big", action="store_true",
                    help="also run a large synthetic Python-only benchmark")
    ap.add_argument("--reps", type=int, default=20,
                    help="Python timing repetitions (default 20)")
    args = ap.parse_args()

    print("=" * 56)
    print("pymatrixeqtl vs R MatrixEQTL -- benchmark")
    print("=" * 56)

    py_time, py_results = time_python(reps=args.reps)
    print("\nPython per-run wall time (example data, cis+trans):")
    for name, t in py_time.items():
        print(f"  {name:>8}: {t * 1000:8.3f} ms")
    print(f"  {'TOTAL':>8}: {sum(py_time.values()) * 1000:8.3f} ms")

    with tempfile.TemporaryDirectory() as td:
        r_dir = Path(td)
        r_elapsed, ok = run_r_reference(r_dir)
        if ok:
            print(f"\nR MatrixEQTL wall time (all 3 models, cis+trans+all): "
                  f"{r_elapsed:.3f} s")
            py_total = sum(py_time.values())
            print(f"Python wall time (3 models, cis+trans):              "
                  f"{py_total:.3f} s")
            if py_total > 0:
                print(f"Speed-up vs R driver process: ~{r_elapsed / py_total:.1f}x "
                      f"(note: R figure includes process + I/O overhead)")
            accuracy(py_results, r_dir)
        else:
            print("\nR / MatrixEQTL not available -- skipping R comparison.")

    if args.big:
        big_benchmark()


if __name__ == "__main__":
    main()
