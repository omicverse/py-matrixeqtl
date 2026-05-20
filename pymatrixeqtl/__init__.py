"""pymatrixeqtl: Pure-Python port of CRAN MatrixEQTL.

A standalone, dependency-light (numpy / scipy / pandas only) port of
*MatrixEQTL* -- Andrey A. Shabalin's ultra-fast eQTL mapping engine
(Shabalin, *Bioinformatics* 2012, 28(10):1353-1358).  It reproduces the
upstream computation bit-for-bit: covariates are orthonormalised, the
expression and genotype matrices are residualised once, and every
SNP-gene test statistic is then a single entry of a big matrix product.

Models
------
* :data:`modelLINEAR`       -- additive linear model (slope, t-stat, p).
* :data:`modelANOVA`        -- genotype as a categorical factor (F-test).
* :data:`modelLINEAR_CROSS` -- linear model with a SNP x last-covariate
  interaction term.

Core data structure
-------------------
* :class:`SlicedData` -- MatrixEQTL's chunked, NumPy-backed matrix
  container for genotype / expression / covariate data.  Supports
  ``create_from_matrix``, ``load_file``, ``row_standardize_centered``,
  ``column_subsample``, ``row_reorder``, ``reslice_combined``,
  ``find_row``, ``set_nan_row_mean`` etc., with R-style method aliases.

Engine
------
* :func:`Matrix_eQTL_main`   -- the full engine: cis/trans splitting by
  genomic distance, Benjamini-Hochberg FDR (computed exactly as upstream),
  optional error-covariance whitening, p-value histograms and the
  ``min.pv.by.genesnp`` tables.
* :func:`Matrix_eQTL_engine` -- thin trans-only wrapper.
* :class:`MatrixEQTLResult`  -- result object with ``.all`` / ``.cis`` /
  ``.trans`` tidy DataFrames (``snps, gene, beta, statistic, pvalue, FDR``),
  ``.param`` and ``.time_in_sec``.

Convenience
-----------
* :func:`eqtl`         -- one-call wrapper accepting paths / arrays /
  DataFrames.
* :func:`load_example` -- MatrixEQTL's own bundled example dataset
  (``SNP.txt``, ``GE.txt``, ``Covariates.txt``, ``geneloc.txt``,
  ``snpsloc.txt``), shipped inside the package.
* :func:`plot_matrix_eqtl` -- p-value histogram / Q-Q plot.

Quick-start
-----------
>>> import pymatrixeqtl as me
>>> ex = me.load_example()
>>> res = me.Matrix_eQTL_main(
...     snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
...     pvOutputThreshold=1e-2, pvOutputThreshold_cis=1e-1,
...     snpspos=ex["snpspos"], genepos=ex["genepos"],
...     useModel=me.modelLINEAR, cisDist=1e6)
>>> res.cis.head()
"""
from __future__ import annotations

from .engine import Matrix_eQTL_engine, Matrix_eQTL_main
from .highlevel import EXAMPLE_DIR, eqtl, load_example
from .models import MODEL_NAMES, modelANOVA, modelLINEAR, modelLINEAR_CROSS
from .plotting import plot_matrix_eqtl
from .results import MatrixEQTLResult
from .sliced_data import SlicedData

__version__ = "0.1.0"

__all__ = [
    # data structure
    "SlicedData",
    # engine
    "Matrix_eQTL_main",
    "Matrix_eQTL_engine",
    "MatrixEQTLResult",
    # models
    "modelLINEAR",
    "modelANOVA",
    "modelLINEAR_CROSS",
    "MODEL_NAMES",
    # convenience
    "eqtl",
    "load_example",
    "EXAMPLE_DIR",
    # plotting
    "plot_matrix_eqtl",
]
