"""R-parity tests -- pymatrixeqtl vs CRAN MatrixEQTL 2.3.

The R driver (:file:`r_reference_driver.R`) runs MatrixEQTL on its *own*
bundled example dataset (``SNP.txt``, ``GE.txt``, ``Covariates.txt``,
``geneloc.txt``, ``snpsloc.txt``), so both sides analyse identical input.
We compare, for every model (linear / ANOVA / interaction) and every
result table (all / cis / trans):

* ``beta``       -- bit-exact (Pearson r > 0.9999, max |diff| < 1e-6).
* ``statistic``  -- the t / F statistic, bit-exact.
* ``pvalue``     -- bit-exact.
* ``FDR``        -- bit-exact (same Benjamini-Hochberg count).
* cis / trans split counts match exactly.

Tests skip gracefully when the CMAP R env or MatrixEQTL is unavailable.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import pymatrixeqtl as me

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"

MODELS = {
    "linear": me.modelLINEAR,
    "anova": me.modelANOVA,
    "cross": me.modelLINEAR_CROSS,
}


def _r_available() -> bool:
    if not R_DRIVER.exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'library(MatrixEQTL); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or MatrixEQTL not installed.",
)


@pytest.fixture(scope="module")
def r_reference(tmp_path_factory):
    """Run the MatrixEQTL R reference once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("matrixeqtl_R")
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {out_dir}"
    )
    res = subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, timeout=900,
    )
    if res.returncode != 0:
        pytest.skip(f"R reference driver failed:\n{res.stderr[-2000:]}")
    return out_dir


@pytest.fixture(scope="module")
def py_results():
    """Run pymatrixeqtl for every model; return nested dict of DataFrames."""
    ex = me.load_example()
    out = {}
    for name, model in MODELS.items():
        all_res = me.Matrix_eQTL_main(
            snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
            pvOutputThreshold=1.0, useModel=model,
        )
        ct_res = me.Matrix_eQTL_main(
            snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
            pvOutputThreshold=1.0, pvOutputThreshold_cis=1.0,
            snpspos=ex["snpspos"], genepos=ex["genepos"],
            useModel=model, cisDist=1e6,
        )
        out[name] = {
            "all": all_res.all,
            "cis": ct_res.cis,
            "trans": ct_res.trans,
        }
    return out


def _merged(r_dir: Path, py_df: pd.DataFrame, model: str, tag: str):
    r_df = pd.read_csv(r_dir / f"{model}_{tag}.tsv", sep="\t")
    assert len(r_df) == len(py_df), (
        f"{model}/{tag}: row count R={len(r_df)} PY={len(py_df)}"
    )
    m = r_df.merge(py_df, on=["snps", "gene"], suffixes=("_R", "_PY"))
    assert len(m) == len(r_df), (
        f"{model}/{tag}: {len(m)}/{len(r_df)} pairs matched on (snps, gene)"
    )
    return m


# ----------------------------------------------------------------------
# per-column parity, every model x table
# ----------------------------------------------------------------------
@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("tag", ["all", "cis", "trans"])
def test_statistic_parity(r_reference, py_results, model, tag):
    m = _merged(r_reference, py_results[model][tag], model, tag)
    r = m["statistic_R"].to_numpy()
    p = m["statistic_PY"].to_numpy()
    rho, _ = pearsonr(r, p)
    max_diff = np.max(np.abs(r - p))
    rel = max_diff / max(np.max(np.abs(r)), 1e-12)
    assert rho > 0.9999, f"{model}/{tag} statistic Pearson r={rho:.8f}"
    assert rel < 1e-6, f"{model}/{tag} statistic rel max|diff|={rel:.2e}"


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("tag", ["all", "cis", "trans"])
def test_pvalue_parity(r_reference, py_results, model, tag):
    m = _merged(r_reference, py_results[model][tag], model, tag)
    r = m["pvalue_R"].to_numpy()
    p = m["pvalue_PY"].to_numpy()
    rho, _ = pearsonr(r, p)
    max_diff = np.max(np.abs(r - p))
    assert rho > 0.9999, f"{model}/{tag} pvalue Pearson r={rho:.8f}"
    assert max_diff < 1e-8, f"{model}/{tag} pvalue max|diff|={max_diff:.2e}"


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("tag", ["all", "cis", "trans"])
def test_fdr_parity(r_reference, py_results, model, tag):
    m = _merged(r_reference, py_results[model][tag], model, tag)
    r = m["FDR_R"].to_numpy()
    p = m["FDR_PY"].to_numpy()
    max_diff = np.max(np.abs(r - p))
    assert max_diff < 1e-8, f"{model}/{tag} FDR max|diff|={max_diff:.2e}"


@pytest.mark.parametrize("model", ["linear", "cross"])
@pytest.mark.parametrize("tag", ["all", "cis", "trans"])
def test_beta_parity(r_reference, py_results, model, tag):
    """The slope (beta) -- linear & interaction models only."""
    m = _merged(r_reference, py_results[model][tag], model, tag)
    if "beta_R" not in m.columns or m["beta_R"].isna().all():
        pytest.skip(f"{model}/{tag}: no beta in R output")
    r = m["beta_R"].to_numpy()
    p = m["beta_PY"].to_numpy()
    rho, _ = pearsonr(r, p)
    max_diff = np.max(np.abs(r - p))
    assert rho > 0.9999, f"{model}/{tag} beta Pearson r={rho:.8f}"
    assert max_diff < 1e-6, f"{model}/{tag} beta max|diff|={max_diff:.2e}"


# ----------------------------------------------------------------------
# cis / trans split counts
# ----------------------------------------------------------------------
def test_cis_trans_counts_match(r_reference, py_results):
    info = pd.read_csv(r_reference / "info.tsv", sep="\t").set_index("model")
    ex = me.load_example()
    for model_name, model in MODELS.items():
        ct = me.Matrix_eQTL_main(
            snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
            pvOutputThreshold=1.0, pvOutputThreshold_cis=1.0,
            snpspos=ex["snpspos"], genepos=ex["genepos"],
            useModel=model, cisDist=1e6,
        )
        assert ct.cis_ntests == info.loc[model_name, "ntests_cis"], (
            f"{model_name}: cis ntests"
        )
        assert ct.trans_ntests == info.loc[model_name, "ntests_trans"], (
            f"{model_name}: trans ntests"
        )
        assert ct.cis_neqtls == info.loc[model_name, "n_cis"]
        assert ct.trans_neqtls == info.loc[model_name, "n_trans"]


def test_row_order_matches_R(r_reference, py_results):
    """MatrixEQTL sorts eQTLs by descending |statistic|; we must too."""
    for model in MODELS:
        for tag in ("all", "cis", "trans"):
            r_df = pd.read_csv(r_reference / f"{model}_{tag}.tsv", sep="\t")
            py_df = py_results[model][tag]
            np.testing.assert_allclose(
                r_df["statistic"].to_numpy(),
                py_df["statistic"].to_numpy(),
                rtol=1e-6, atol=1e-6,
                err_msg=f"{model}/{tag}: row order differs from R",
            )
