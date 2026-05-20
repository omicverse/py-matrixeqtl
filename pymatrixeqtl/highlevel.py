"""High-level convenience API and example-data loader for :mod:`pymatrixeqtl`.

* :func:`eqtl` -- a one-call wrapper that accepts file paths, NumPy arrays
  or :class:`pandas.DataFrame` objects and runs the full engine.
* :func:`load_example` -- loads MatrixEQTL's own bundled example dataset
  (``SNP.txt``, ``GE.txt``, ``Covariates.txt``, ``geneloc.txt``,
  ``snpsloc.txt``).  A copy of those five files ships inside this package
  so the example works without R installed.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import pandas as pd

from .engine import Matrix_eQTL_main
from .models import modelLINEAR
from .results import MatrixEQTLResult
from .sliced_data import SlicedData

__all__ = ["eqtl", "load_example", "EXAMPLE_DIR"]

#: Directory holding the bundled MatrixEQTL example data.
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data")

_MatrixInput = Union[str, np.ndarray, pd.DataFrame, SlicedData]


def _as_sliced(obj: Optional[_MatrixInput], delimiter: str = "\t") -> SlicedData:
    """Coerce a file path / array / DataFrame / SlicedData to SlicedData."""
    if obj is None:
        return SlicedData()
    if isinstance(obj, SlicedData):
        return obj
    if isinstance(obj, str):
        sd = SlicedData()
        sd.load_file(obj, delimiter=delimiter)
        return sd
    return SlicedData(obj)


def eqtl(
    snps: _MatrixInput,
    gene: _MatrixInput,
    cvrt: Optional[_MatrixInput] = None,
    *,
    model: int = modelLINEAR,
    pv_threshold: float = 1e-5,
    pv_threshold_cis: float = 0.0,
    snpspos: Optional[Union[str, pd.DataFrame]] = None,
    genepos: Optional[Union[str, pd.DataFrame]] = None,
    cis_dist: float = 1e6,
    error_covariance=None,
    min_pv_by_genesnp: bool = False,
    pvalue_hist=False,
    verbose: bool = False,
    output_file_name: Optional[str] = None,
    output_file_name_cis: Optional[str] = None,
    n_anova_groups: int = 3,
) -> MatrixEQTLResult:
    """Run an eQTL analysis -- friendly wrapper around :func:`Matrix_eQTL_main`.

    Parameters
    ----------
    snps, gene, cvrt :
        Genotype, expression and covariate inputs.  Each may be a TSV path,
        a 2-D NumPy array, a :class:`pandas.DataFrame` (index = features,
        columns = samples) or a :class:`~pymatrixeqtl.SlicedData`.
    model :
        ``modelLINEAR`` (default), ``modelANOVA`` or ``modelLINEAR_CROSS``.
    pv_threshold, pv_threshold_cis :
        Trans / cis p-value reporting thresholds.
    snpspos, genepos :
        Position tables (paths or DataFrames) enabling the cis/trans split.
    cis_dist :
        Maximum SNP-gene distance (bp) for a cis association.
    error_covariance :
        Optional error-covariance matrix for whitening.
    min_pv_by_genesnp, pvalue_hist, verbose :
        Passed through to the engine.
    output_file_name, output_file_name_cis :
        Optional TSV output paths.
    n_anova_groups :
        Genotype-category count for the ANOVA model.

    Returns
    -------
    MatrixEQTLResult
    """
    snps_sd = _as_sliced(snps)
    gene_sd = _as_sliced(gene)
    cvrt_sd = _as_sliced(cvrt)

    snpspos_df = (
        pd.read_csv(snpspos, sep="\t") if isinstance(snpspos, str) else snpspos
    )
    genepos_df = (
        pd.read_csv(genepos, sep="\t") if isinstance(genepos, str) else genepos
    )

    return Matrix_eQTL_main(
        snps=snps_sd,
        gene=gene_sd,
        cvrt=cvrt_sd,
        output_file_name=output_file_name,
        pvOutputThreshold=pv_threshold,
        useModel=model,
        errorCovariance=error_covariance,
        verbose=verbose,
        output_file_name_cis=output_file_name_cis,
        pvOutputThreshold_cis=pv_threshold_cis,
        snpspos=snpspos_df,
        genepos=genepos_df,
        cisDist=cis_dist,
        pvalue_hist=pvalue_hist,
        min_pv_by_genesnp=min_pv_by_genesnp,
        n_anova_groups=n_anova_groups,
    )


def load_example() -> dict:
    """Load MatrixEQTL's bundled example dataset.

    Returns
    -------
    dict
        Keys ``snps`` / ``gene`` / ``cvrt`` (:class:`SlicedData`) and
        ``snpspos`` / ``genepos`` (:class:`pandas.DataFrame`), plus the
        file paths under ``paths``.
    """
    paths = {
        "snps": os.path.join(EXAMPLE_DIR, "SNP.txt"),
        "gene": os.path.join(EXAMPLE_DIR, "GE.txt"),
        "cvrt": os.path.join(EXAMPLE_DIR, "Covariates.txt"),
        "snpspos": os.path.join(EXAMPLE_DIR, "snpsloc.txt"),
        "genepos": os.path.join(EXAMPLE_DIR, "geneloc.txt"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Bundled example data not found: {missing}"
        )
    snps = SlicedData()
    snps.load_file(paths["snps"])
    gene = SlicedData()
    gene.load_file(paths["gene"])
    cvrt = SlicedData()
    cvrt.load_file(paths["cvrt"])
    return {
        "snps": snps,
        "gene": gene,
        "cvrt": cvrt,
        "snpspos": pd.read_csv(paths["snpspos"], sep="\t"),
        "genepos": pd.read_csv(paths["genepos"], sep="\t"),
        "paths": paths,
    }
