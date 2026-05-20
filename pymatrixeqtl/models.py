"""eQTL model definitions for :mod:`pymatrixeqtl`.

MatrixEQTL distinguishes three regression models, identified in R by magic
integer constants.  We keep the *exact* same integers so the values are
interchangeable with R code and stable across versions:

================  =========  ====================================
Constant          Integer    Test
================  =========  ====================================
``modelLINEAR``   117348     additive linear: expr ~ SNP + cvrt
``modelANOVA``    47074      genotype as a categorical factor (F-test)
``modelLINEAR_CROSS`` 1113461  linear with SNP x last-covariate term
================  =========  ====================================
"""
from __future__ import annotations

__all__ = ["modelLINEAR", "modelANOVA", "modelLINEAR_CROSS", "MODEL_NAMES"]

# Magic constants -- identical to MatrixEQTL's R values.
modelLINEAR: int = 117348
modelANOVA: int = 47074
modelLINEAR_CROSS: int = 1113461

#: Human-readable name for each model constant.
MODEL_NAMES = {
    modelLINEAR: "modelLINEAR",
    modelANOVA: "modelANOVA",
    modelLINEAR_CROSS: "modelLINEAR_CROSS",
}
