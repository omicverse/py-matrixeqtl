"""The MatrixEQTL engine -- :func:`Matrix_eQTL_main` and friends.

This module is a faithful, line-by-line Python port of
``R/Matrix_eQTL_engine.r`` from the CRAN package *MatrixEQTL* 2.3
(Shabalin, *Bioinformatics* 2012).  It reproduces MatrixEQTL's fast eQTL
mapping exactly:

1. Covariates are augmented with an intercept and orthonormalised (QR).
2. Expression is mean-imputed, residualised on the covariates and each
   row scaled to unit norm.
3. For every slice of SNPs the genotype is processed (additive numeric,
   ANOVA dummy split, or cross interaction), residualised, orthonormalised,
   and the test statistic for every SNP-gene pair is then just an entry of
   the matrix product ``gene_residual %*% t(snp_residual)`` -- a
   correlation.  The t / F statistic and p-value follow analytically.
4. cis / trans associations are split by genomic distance; Benjamini-
   Hochberg FDR is computed over *all* tested pairs (not just those that
   pass the threshold), exactly as MatrixEQTL does.

The public entry points are :func:`Matrix_eQTL_main` and the thin wrapper
:func:`Matrix_eQTL_engine`.  Both return a :class:`MatrixEQTLResult`.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .models import MODEL_NAMES, modelANOVA, modelLINEAR, modelLINEAR_CROSS
from .results import MatrixEQTLResult
from .sliced_data import SlicedData

__all__ = ["Matrix_eQTL_main", "Matrix_eQTL_engine"]

_EPS = np.finfo(np.float64).eps
_XMIN = np.finfo(np.float64).tiny  # .Machine$double.xmin


# ----------------------------------------------------------------------
# small helpers mirroring MatrixEQTL internals
# ----------------------------------------------------------------------
def _set_nan_row_mean(x: np.ndarray) -> np.ndarray:
    """Impute NaNs with row means (R ``.SetNanRowMean``)."""
    if not np.isnan(x).any():
        return x
    x = x.copy()
    with np.errstate(invalid="ignore"):
        rowmean = np.nanmean(x, axis=1)
    rowmean[np.isnan(rowmean)] = 0.0
    inds = np.where(np.isnan(x).any(axis=1))[0]
    for j in inds:
        mask = np.isnan(x[j, :])
        x[j, mask] = rowmean[j]
    return x


def _pv_nz(x: np.ndarray) -> np.ndarray:
    """Clamp p-values away from exact zero (R ``.pv.nz``)."""
    return np.maximum(x, _XMIN)


def _snp_process_split_for_anova(x: np.ndarray, n_groups: int):
    """Split a genotype matrix into ANOVA dummy matrices.

    Faithful port of ``.SNP_process_split_for_ANOVA``.  ``x`` is
    ``nSnps x nSamples``; the result is a list of ``n_groups - 1`` boolean
    matrices, one per non-modal genotype category.
    """
    flat = x.ravel()
    uniq = np.unique(flat[~np.isnan(flat)])
    uniq = list(uniq)
    if len(uniq) > n_groups:
        raise ValueError(
            "More than declared number of genotype categories "
            "is detected in ANOVA"
        )
    elif len(uniq) < n_groups:
        uniq = uniq + [min(uniq) - 1] * (n_groups - len(uniq))
    uniq = np.array(uniq, dtype=np.float64)

    nr = x.shape[0]
    freq = np.zeros((nr, n_groups))
    for i in range(n_groups):
        freq[:, i] = np.nansum(x == uniq[i], axis=1)

    x = x.copy()
    x[np.isnan(x)] = float(np.min(uniq) - 2)

    rez = []
    # R: md = which.max(freq); freq[,md] = -1   (drop the modal category)
    md = np.argmax(freq, axis=1)
    freq[np.arange(nr), md] = -1
    for _ in range(n_groups - 1):
        md = np.argmax(freq, axis=1)
        freq[np.arange(nr), md] = -1
        rez.append((x == uniq[md][:, None]).astype(np.float64))
    return rez


def _array_ind(linear, shape):
    """Convert column-major (R/Fortran) linear indices to (row, col)."""
    nr = shape[0]
    rows = linear % nr
    cols = linear // nr
    return rows, cols


# ----------------------------------------------------------------------
# output saver -- accumulates significant pairs, computes FDR
# ----------------------------------------------------------------------
class _OutputSaverFRD:
    """Port of MatrixEQTL's ``.OutputSaver_FRD`` reference class.

    Accumulates (snp, gene, statistic, beta) tuples, then in
    :meth:`get_results` sorts by absolute statistic, converts to t/F and
    p-values and applies the Benjamini-Hochberg correction over
    ``fdr_total_count`` tests.
    """

    def __init__(self) -> None:
        self.spos: list = []
        self.gpos: list = []
        self.stat: list = []
        self.beta: list = []
        self.testfun = None
        self.pvfun = None

    def start(self, testfun, pvfun) -> None:
        self.testfun = testfun
        self.pvfun = pvfun

    def update(self, spos, gpos, sta, beta=None) -> None:
        if sta.size > 0:
            self.spos.append(np.asarray(spos))
            self.gpos.append(np.asarray(gpos))
            self.stat.append(np.asarray(sta))
            if beta is not None:
                self.beta.append(np.asarray(beta))

    def get_results(self, gene: SlicedData, snps: SlicedData, fdr_total_count: int):
        """Return the tidy eQTL DataFrame with FDR (R ``getResults``)."""
        if not self.stat:
            return pd.DataFrame(
                {
                    "snps": pd.Series([], dtype=object),
                    "gene": pd.Series([], dtype=object),
                    "beta": pd.Series([], dtype=float),
                    "statistic": pd.Series([], dtype=float),
                    "pvalue": pd.Series([], dtype=float),
                    "FDR": pd.Series([], dtype=float),
                }
            )
        cdata = np.concatenate(self.stat)
        sdata = np.concatenate(self.spos)
        gdata = np.concatenate(self.gpos)
        tests = self.testfun(cdata)
        # R: sort.list(abs(tests), decreasing = TRUE)
        ordr = np.argsort(-np.abs(tests), kind="stable")
        tests = tests[ordr]
        pvalues = self.pvfun(tests)
        n = len(pvalues)
        fdr = pvalues * fdr_total_count / np.arange(1, n + 1)
        fdr[-1] = min(fdr[-1], 1.0)
        # rev(cummin(rev(FDR)))
        fdr = np.minimum.accumulate(fdr[::-1])[::-1].copy()
        snp_names = snps.row_names[sdata[ordr]]
        gene_names = gene.row_names[gdata[ordr]]
        out = {
            "snps": snp_names,
            "gene": gene_names,
            "statistic": tests,
            "pvalue": pvalues,
            "FDR": fdr,
        }
        if self.beta:
            beta = np.concatenate(self.beta)[ordr]
            out["beta"] = beta
            out = {
                "snps": out["snps"],
                "gene": out["gene"],
                "beta": out["beta"],
                "statistic": out["statistic"],
                "pvalue": out["pvalue"],
                "FDR": out["FDR"],
            }
        return pd.DataFrame(out)


class _MinPValue:
    """Port of MatrixEQTL's ``.minpvalue`` -- best p-value per SNP / gene."""

    def __init__(self, snps: SlicedData, gene: SlicedData) -> None:
        self.sdata = [np.zeros(snps.get_n_rows_in_slice(s + 1))
                      for s in range(snps.n_slices())]
        self.gdata = [np.zeros(gene.get_n_rows_in_slice(g + 1))
                      for g in range(gene.n_slices())]

    def update(self, ss: int, gg: int, astat: np.ndarray) -> None:
        # astat: nGenes x nSnps  (rows = genes in slice gg, cols = snps in ss)
        gmax = self.gdata[gg - 1]
        self.gdata[gg - 1] = np.maximum(gmax, astat.max(axis=1))
        smax = self.sdata[ss - 1]
        self.sdata[ss - 1] = np.maximum(smax, astat.max(axis=0))

    def get_results(self, snps: SlicedData, gene: SlicedData, pvfun):
        min_pv_snps = pd.Series(
            pvfun(np.concatenate(self.sdata)) if self.sdata else np.array([]),
            index=snps.row_names,
        )
        min_pv_gene = pd.Series(
            pvfun(np.concatenate(self.gdata)) if self.gdata else np.array([]),
            index=gene.row_names,
        )
        return min_pv_snps, min_pv_gene


class _Histogrammer:
    """Port of MatrixEQTL's ``.histogrammer`` -- p-value histogram bins."""

    def __init__(self, pvbins: np.ndarray, statbins: np.ndarray) -> None:
        ordr = np.argsort(statbins, kind="stable")
        self.pvbins = np.asarray(pvbins)[ordr]
        self.statbins = np.asarray(statbins, dtype=np.float64)[ordr]
        self.statbins[-1] = np.finfo(np.float64).max
        self.count = np.zeros(len(pvbins) - 1)

    def update(self, stats_for_hist: np.ndarray) -> None:
        vals = np.asarray(stats_for_hist).ravel()
        idx = np.searchsorted(self.statbins, vals, side="right")
        t = np.bincount(idx, minlength=len(self.statbins) + 1)
        # findInterval -> bins 1..len(statbins)-1
        self.count = self.count + t[1:len(self.statbins)]

    def get_results(self):
        if np.all(np.diff(self.pvbins) >= 0):
            return self.pvbins, self.count
        return self.pvbins[::-1], self.count[::-1]


# ----------------------------------------------------------------------
# model factory: thresholding / statistic / p-value functions
# ----------------------------------------------------------------------
def _make_model(use_model, n_samples, n_cov, n_anova_groups, gene_std, snps_std,
                pv_thr, pv_thr_cis):
    """Build the per-model statistic / p-value / threshold closures.

    Returns a dict mirroring the variables MatrixEQTL defines inside the
    ``useModel == ...`` branches.
    """
    if use_model == modelLINEAR:
        n_var_tested = 1
        df_full = n_samples - n_cov - n_var_tested

        def snps_process(x):
            return [_set_nan_row_mean(x)]

        def statistic_fun(mat_list):
            return mat_list[0]

        def afun(x):
            return np.abs(x)

        def threshfun(pv):
            if pv <= 0:
                return 1.0
            if pv >= 1:
                return 0.0
            thr = stats.t.isf(pv / 2.0, df_full)
            thr = thr ** 2
            return float(np.sqrt(thr / (df_full + thr)))

        def testfun(x):
            return x * np.sqrt(df_full / (1.0 - np.minimum(x ** 2, 1.0)))

        def pvfun(x):
            return _pv_nz(stats.t.cdf(-np.abs(x), df_full) * 2.0)

        def betafun(stat, ss, gg, sel_rows, sel_cols):
            return stat * gene_std[gg][sel_rows] / snps_std[ss][sel_cols]

        statistic_name = "beta\tt-stat"

    elif use_model == modelANOVA:
        n_var_tested = n_anova_groups - 1
        df_full = n_samples - n_cov - n_var_tested

        def snps_process(x):
            return _snp_process_split_for_anova(x, n_anova_groups)

        def statistic_fun(mat_list):
            x = mat_list[0] ** 2
            for j in range(1, len(mat_list)):
                x = x + mat_list[j] ** 2
            return x

        def afun(x):
            return x

        def threshfun(pv):
            if pv <= 0:
                return 1.0
            if pv >= 1:
                return 0.0
            thr = stats.f.isf(pv, n_var_tested, df_full)
            return float(thr / (df_full / n_var_tested + thr))

        def testfun(x):
            return x / (1.0 - np.minimum(x, 1.0)) * (df_full / n_var_tested)

        def pvfun(x):
            return _pv_nz(stats.f.sf(x, n_var_tested, df_full))

        betafun = None
        statistic_name = "F-test"

    elif use_model == modelLINEAR_CROSS:
        n_var_tested = 1
        df_full = n_samples - n_cov - n_var_tested - 1

        def statistic_fun(mat_list):
            return mat_list[1] / np.sqrt(1.0 - mat_list[0] ** 2)

        def afun(x):
            return np.abs(x)

        def threshfun(pv):
            if pv <= 0:
                return 1.0
            if pv >= 1:
                return 0.0
            thr = stats.t.isf(pv / 2.0, df_full)
            thr = thr ** 2
            return float(np.sqrt(thr / (df_full + thr)))

        def testfun(x):
            return x * np.sqrt(df_full / (1.0 - np.minimum(x ** 2, 1.0)))

        def pvfun(x):
            return _pv_nz(stats.t.cdf(-np.abs(x), df_full) * 2.0)

        def betafun(stat, ss, gg, sel_rows, sel_cols):
            return stat * gene_std[gg][sel_rows] / snps_std[ss][sel_cols]

        statistic_name = "beta\tt-stat"
        snps_process = None  # set by caller (needs last.covariate)
    else:  # pragma: no cover
        raise ValueError(f"Unknown model {use_model}")

    return {
        "n_var_tested": n_var_tested,
        "df_full": df_full,
        "snps_process": snps_process if use_model != modelLINEAR_CROSS else None,
        "statistic_fun": statistic_fun,
        "afun": afun,
        "threshfun": threshfun,
        "testfun": testfun,
        "pvfun": pvfun,
        "betafun": betafun,
        "statistic_name": statistic_name,
        "thresh": threshfun(pv_thr),
        "thresh_cis": threshfun(pv_thr_cis),
    }


# ----------------------------------------------------------------------
# cis/trans position machinery
# ----------------------------------------------------------------------
def _build_positions(gene, snps, snpspos, genepos, cis_dist):
    """Compute linearised genomic positions for SNPs and genes.

    Faithful port of the ``pvOutputThreshold.cis > 0`` block in
    ``Matrix_eQTL_main``.  Returns ``(snps_pos, gene_pos)`` -- 1-D and 2-D
    arrays aligned to the (reordered) SNP / gene rows -- after reordering
    ``snps`` and ``gene`` in place so positions are sorted.
    """
    gene_names = gene.row_names
    snps_names = snps.row_names

    genepos = genepos.copy()
    gp3 = genepos.iloc[:, 2].to_numpy(dtype=np.float64)
    gp4 = genepos.iloc[:, 3].to_numpy(dtype=np.float64)
    if np.any(gp3 > gp4):
        lo = np.minimum(gp3, gp4)
        hi = np.maximum(gp3, gp4)
        gp3, gp4 = lo, hi

    gene_id = genepos.iloc[:, 0].astype(str).to_numpy()
    snp_id = snpspos.iloc[:, 0].astype(str).to_numpy()
    gene_chr_raw = genepos.iloc[:, 1].astype(str).to_numpy()
    snp_chr_raw = snpspos.iloc[:, 1].astype(str).to_numpy()
    snp_pos_raw = snpspos.iloc[:, 2].to_numpy(dtype=np.float64)

    # match() against position tables
    gene_lookup = {g: i for i, g in enumerate(gene_id)}
    snp_lookup = {s: i for i, s in enumerate(snp_id)}
    genematch = np.array([gene_lookup.get(str(g), -1) for g in gene_names])
    snpsmatch = np.array([snp_lookup.get(str(s), -1) for s in snps_names])

    used_gene = np.zeros(len(gene_id), dtype=bool)
    used_gene[genematch[genematch >= 0]] = True
    used_snps = np.zeros(len(snp_id), dtype=bool)
    used_snps[snpsmatch[snpsmatch >= 0]] = True
    if not used_gene.any():
        raise ValueError("Gene names do not match those in the gene location file.")
    if not used_snps.any():
        raise ValueError("SNP names do not match those in the SNP location file.")

    # chromosome ordering: numeric first (sorted), then non-numeric
    chr_names = []
    seen = set()
    for c in list(snp_chr_raw[used_snps]) + list(gene_chr_raw[used_gene]):
        if c not in seen:
            seen.add(c)
            chr_names.append(c)

    def _as_int(c):
        try:
            return int(c)
        except (ValueError, TypeError):
            return None

    numeric = sorted((c for c in chr_names if _as_int(c) is not None),
                     key=_as_int)
    non_numeric = [c for c in chr_names if _as_int(c) is None]
    chr_names = numeric + non_numeric
    chr_index = {c: i for i, c in enumerate(chr_names)}

    gene_chr = np.array([chr_index.get(c, np.nan) for c in gene_chr_raw])
    snps_chr = np.array([chr_index.get(c, np.nan) for c in snp_chr_raw])

    chr_max = (
        max(np.max(snp_pos_raw[used_snps]), np.max(gp4[used_gene])) + cis_dist
    )

    genepos2 = np.column_stack([gp3, gp4]) + (gene_chr[:, None] - 1) * chr_max
    snpspos2 = snp_pos_raw + (snps_chr - 1) * chr_max

    snps_pos = np.zeros(len(snps_names))
    has = snpsmatch >= 0
    snps_pos[has] = snpspos2[snpsmatch[has]]
    snps_pos[np.isnan(snps_pos)] = 0.0
    snps_pos[snps_pos == 0] = (len(chr_names) + 1) * (chr_max + cis_dist)

    gene_pos = np.zeros((len(gene_names), 2))
    hasg = genematch >= 0
    gene_pos[hasg] = genepos2[genematch[hasg]]
    bad = np.isnan(gene_pos).any(axis=1)
    gene_pos[bad] = 0.0
    gene_pos[(gene_pos == 0).any(axis=1)] = (len(chr_names) + 2) * (chr_max + cis_dist)

    # reorder SNPs / genes so positions are sorted
    if np.any(np.diff(snps_pos) < 0):
        ordr = np.argsort(snps_pos, kind="stable")
        snps.row_reorder(ordr)
        snps_pos = snps_pos[ordr]
    grow = gene_pos.sum(axis=1)
    if np.any(np.diff(grow) < 0):
        ordr = np.argsort(grow, kind="stable")
        gene.row_reorder(ordr)
        gene_pos = gene_pos[ordr]
    return snps_pos, gene_pos


# ----------------------------------------------------------------------
# main engine
# ----------------------------------------------------------------------
def Matrix_eQTL_main(  # noqa: N802 -- R-compatible name
    snps: SlicedData,
    gene: SlicedData,
    cvrt: Optional[SlicedData] = None,
    output_file_name: Optional[str] = None,
    pvOutputThreshold: float = 1e-5,
    useModel: int = modelLINEAR,
    errorCovariance=None,
    verbose: bool = False,
    output_file_name_cis: Optional[str] = None,
    pvOutputThreshold_cis: float = 0.0,
    snpspos: Optional[pd.DataFrame] = None,
    genepos: Optional[pd.DataFrame] = None,
    cisDist: float = 1e6,
    pvalue_hist=False,
    min_pv_by_genesnp: bool = False,
    noFDRsaveMemory: bool = False,
    n_anova_groups: int = 3,
) -> MatrixEQTLResult:
    """Run a fast eQTL analysis -- port of R ``Matrix_eQTL_main``.

    Parameters
    ----------
    snps, gene, cvrt :
        :class:`~pymatrixeqtl.SlicedData` objects holding the genotype,
        expression and (optional) covariate matrices; columns are samples.
    output_file_name, output_file_name_cis :
        If given, the trans / cis result tables are also written as TSV.
    pvOutputThreshold :
        p-value threshold for *trans* (or all) associations.  Set to 0 to
        skip the trans pass when doing cis-only.
    useModel :
        One of :data:`~pymatrixeqtl.modelLINEAR`,
        :data:`~pymatrixeqtl.modelANOVA`,
        :data:`~pymatrixeqtl.modelLINEAR_CROSS`.
    errorCovariance :
        Optional sample-by-sample error covariance matrix; triggers a
        whitening transform.
    output_file_name_cis, pvOutputThreshold_cis, snpspos, genepos, cisDist :
        Enable cis/trans splitting.  ``snpspos`` has columns
        ``[snpid, chr, pos]``; ``genepos`` has ``[geneid, chr, left, right]``.
    pvalue_hist :
        ``False``, ``True``, ``"qqplot"``, or an int bin count -- records a
        p-value histogram for plotting.
    min_pv_by_genesnp :
        Also record the best p-value per SNP and per gene.
    noFDRsaveMemory :
        Do not accumulate results in memory (TSV output only).
    n_anova_groups :
        Number of genotype categories for ANOVA (default 3).

    Returns
    -------
    MatrixEQTLResult
        Object exposing ``.all`` / ``.cis`` / ``.trans`` tidy DataFrames,
        ``.param``, ``.time_in_sec`` and the ``min.pv.by.genesnp`` tables.
    """
    start_time = time.time()

    if cvrt is None:
        cvrt = SlicedData()
    if not isinstance(snps, SlicedData):
        snps = SlicedData(snps)
    if not isinstance(gene, SlicedData):
        gene = SlicedData(gene)
    if not isinstance(cvrt, SlicedData):
        cvrt = SlicedData(cvrt)

    # ---- validation -------------------------------------------------
    if min(snps.n_rows(), snps.n_cols()) == 0:
        raise ValueError("Empty genotype dataset")
    if min(gene.n_rows(), gene.n_cols()) == 0:
        raise ValueError("Empty expression dataset")
    if snps.n_cols() != gene.n_cols():
        raise ValueError(
            "Different number of samples in the genotype and "
            "gene expression files"
        )
    if cvrt.n_rows() > 0 and snps.n_cols() != cvrt.n_cols():
        raise ValueError("Wrong number of samples in the matrix of covariates")
    if not (0 <= pvOutputThreshold <= 1):
        raise ValueError("pvOutputThreshold must be in [0, 1]")
    if not (0 <= pvOutputThreshold_cis <= 1):
        raise ValueError("pvOutputThreshold_cis must be in [0, 1]")
    if not (pvOutputThreshold == 0 or pvOutputThreshold_cis == 0
            or pvOutputThreshold <= pvOutputThreshold_cis):
        raise ValueError("pvOutputThreshold must be <= pvOutputThreshold_cis")
    if not (pvOutputThreshold > 0 or pvOutputThreshold_cis > 0):
        raise ValueError("At least one p-value threshold must be positive")
    if useModel not in (modelLINEAR, modelANOVA, modelLINEAR_CROSS):
        raise ValueError(f"Unknown useModel: {useModel}")
    if useModel in (modelLINEAR, modelLINEAR_CROSS):
        if snps.n_cols() <= cvrt.n_rows() + 1 + 1:
            raise ValueError(
                "The number of covariates exceeds the number of samples. "
                "Linear regression can not be fit."
            )
    if useModel == modelLINEAR_CROSS and cvrt.n_rows() == 0:
        raise ValueError('Model "modelLINEAR_CROSS" requires at least one covariate')
    if useModel == modelANOVA:
        if n_anova_groups != int(n_anova_groups) or n_anova_groups < 3:
            raise ValueError("n_anova_groups must be an integer >= 3")
        if snps.n_cols() <= cvrt.n_rows() + n_anova_groups:
            raise ValueError(
                "The number of covariates exceeds the number of samples. "
                "ANOVA can not be fit."
            )
    if pvOutputThreshold_cis > 0:
        if snpspos is None or genepos is None:
            raise ValueError("snpspos and genepos required for cis analysis")
        if snpspos.shape[1] != 3:
            raise ValueError("snpspos must have 3 columns")
        if genepos.shape[1] != 4:
            raise ValueError("genepos must have 4 columns")

    error_cov = None
    if errorCovariance is not None and np.size(errorCovariance) > 0:
        error_cov = np.asarray(errorCovariance, dtype=np.float64)
        if error_cov.ndim == 1:
            error_cov = error_cov.reshape(1, -1)
        if error_cov.shape[0] != error_cov.shape[1]:
            raise ValueError("The covariance matrix is not square")
        if error_cov.shape[0] != snps.n_cols():
            raise ValueError(
                "The covariance matrix size does not match the number of samples"
            )
        if not np.allclose(error_cov, error_cov.T):
            raise ValueError("The covariance matrix is not symmetric")

    # ---- clone mutable inputs --------------------------------------
    gene = gene.clone()
    cvrt = cvrt.clone()
    snps = snps.clone()

    params = {
        "output_file_name": output_file_name,
        "pvOutputThreshold": pvOutputThreshold,
        "useModel": useModel,
        "useModelName": MODEL_NAMES[useModel],
        "errorCovariance": error_cov,
        "verbose": verbose,
        "output_file_name.cis": output_file_name_cis,
        "pvOutputThreshold.cis": pvOutputThreshold_cis,
        "cisDist": cisDist,
        "pvalue.hist": pvalue_hist,
        "min.pv.by.genesnp": min_pv_by_genesnp,
    }

    def status(text):
        if verbose and text:
            print(text)

    # ---- error covariance whitening --------------------------------
    if error_cov is not None:
        status("Processing the error covariance matrix")
        d, v = np.linalg.eigh(error_cov)
        if np.any(d <= 0):
            raise ValueError("The covariance matrix is not positive definite")
        correction = (v * (1.0 / np.sqrt(d))) @ v.T
    else:
        correction = None

    # ---- cis/trans position matching -------------------------------
    geneloc = None
    snpsloc = None
    gene_pos = None
    snps_pos = None
    if pvOutputThreshold_cis > 0:
        status("Matching data files and location files")
        snps_pos, gene_pos = _build_positions(gene, snps, snpspos, genepos, cisDist)
        # per-slice position lists
        geneloc = []
        off = 0
        for gc in range(gene.n_slices()):
            nr = gene.get_n_rows_in_slice(gc + 1)
            geneloc.append(gene_pos[off:off + nr, :])
            off += nr
        snpsloc = []
        off = 0
        for sc in range(snps.n_slices()):
            nr = snps.get_n_rows_in_slice(sc + 1)
            snpsloc.append(snps_pos[off:off + nr])
            off += nr

    # ---- covariates ------------------------------------------------
    status("Processing covariates")
    last_covariate = None
    if useModel == modelLINEAR_CROSS:
        last_slice = cvrt.get_slice(cvrt.n_slices())
        last_covariate = np.asarray(last_slice[-1, :], dtype=np.float64)

    if cvrt.n_rows() > 0:
        cvrt.set_nan_row_mean()
        cvrt.combine_in_one_slice()
        cvrt_mat = np.vstack([np.ones((1, snps.n_cols())), cvrt.get_slice(1)])
    else:
        cvrt_mat = np.ones((1, snps.n_cols()))

    if correction is not None:
        cvrt_mat = cvrt_mat @ correction

    # qr(t(cvrt)) -> orthonormal rows
    q, r = np.linalg.qr(cvrt_mat.T)
    if np.min(np.abs(np.diag(r))) < _EPS * snps.n_cols():
        raise ValueError("Colinear or zero covariates detected")
    cvrt_mat = q.T  # nCov x nSamples, orthonormal rows
    n_cov = cvrt_mat.shape[0]

    # ---- gene residualisation --------------------------------------
    status("Processing gene expression data (imputation, residualization)")
    gene.set_nan_row_mean()
    if correction is not None:
        gene.row_matrix_multiply(correction)

    gene_std = {}  # 1-based slice -> per-row std divisor
    gene_offsets = [0]
    for sl in range(1, gene.n_slices() + 1):
        slice_ = gene.get_slice(sl).copy()
        gene_offsets.append(gene_offsets[-1] + slice_.shape[0])
        rowsq1 = np.sum(slice_ ** 2, axis=1)
        slice_ = slice_ - (slice_ @ cvrt_mat.T) @ cvrt_mat
        rowsq2 = np.sum(slice_ ** 2, axis=1)
        delete_rows = rowsq2 <= rowsq1 * _EPS
        slice_[delete_rows, :] = 0.0
        rowsq2 = rowsq2.copy()
        rowsq2[delete_rows] = 1.0
        div = np.sqrt(rowsq2)
        gene_std[sl] = div
        gene.set_slice(sl, slice_ / div[:, None])

    n_samples = snps.n_cols()

    # ---- model functions -------------------------------------------
    model = _make_model(useModel, n_samples, n_cov, n_anova_groups,
                        gene_std, {}, pvOutputThreshold, pvOutputThreshold_cis)
    snps_std = {}
    model_snps_std = snps_std

    def betafun(stat, ss, gg, sel_rows, sel_cols):
        if model["betafun"] is None:
            return None
        return stat * gene_std[gg][sel_rows] / model_snps_std[ss][sel_cols]

    # snps_process for each model (needs n_anova_groups / last_covariate)
    if useModel == modelLINEAR:
        def snps_process(x):
            return [_set_nan_row_mean(x)]
    elif useModel == modelANOVA:
        def snps_process(x):
            return _snp_process_split_for_anova(x, n_anova_groups)
    else:  # modelLINEAR_CROSS
        def snps_process(x):
            base = _set_nan_row_mean(x)
            cross = base * last_covariate[None, :]
            return [base, cross]

    thresh = model["thresh"]
    thresh_cis = model["thresh_cis"]
    statistic_fun = model["statistic_fun"]
    afun = model["afun"]
    testfun = model["testfun"]
    pvfun = model["pvfun"]
    threshfun = model["threshfun"]

    # ---- output savers ---------------------------------------------
    saver_tra = None
    saver_cis = None
    if pvOutputThreshold > 0:
        saver_tra = _OutputSaverFRD()
        saver_tra.start(testfun, pvfun)
    if pvOutputThreshold_cis > 0:
        saver_cis = _OutputSaverFRD()
        saver_cis.start(testfun, pvfun)

    # ---- histograms / minpv ----------------------------------------
    pvbins = None
    if pvalue_hist is not False:
        if pvalue_hist == "qqplot":
            log_xmin = np.log10(_XMIN)
            pvbins = np.concatenate(
                [[0.0], 10.0 ** np.arange(log_xmin - 1, 0 + 1e-9, 0.05)]
            )
        elif pvalue_hist is True:
            pvbins = np.linspace(0, 1, 101)
        elif isinstance(pvalue_hist, (int, float)) and not isinstance(pvalue_hist, bool):
            pvbins = np.linspace(0, 1, int(pvalue_hist) + 1)
        elif hasattr(pvalue_hist, "__len__"):
            pvbins = np.asarray(pvalue_hist, dtype=np.float64)
        else:
            raise ValueError('pvalue_hist must be False, True, "qqplot", or numeric')

    do_hist = pvbins is not None
    hist_all = hist_cis = None
    if do_hist:
        pvbins = np.sort(pvbins)
        statbins = np.array([threshfun(p) for p in pvbins])
        if pvOutputThreshold > 0:
            hist_all = _Histogrammer(pvbins, statbins)
        if pvOutputThreshold_cis > 0:
            hist_cis = _Histogrammer(pvbins, statbins)

    minpv_tra = minpv_cis = None
    if min_pv_by_genesnp:
        if pvOutputThreshold > 0:
            minpv_tra = _MinPValue(snps, gene)
        if pvOutputThreshold_cis > 0:
            minpv_cis = _MinPValue(snps, gene)

    # ---- cis interval pre-screen -----------------------------------
    if pvOutputThreshold_cis > 0:
        sn_l = np.array([s[0] for s in snpsloc])
        sn_r = np.array([s[-1] for s in snpsloc])
        ge_l = np.array([g.min() for g in geneloc])
        ge_r = np.array([g.max() for g in geneloc])
        ge_l = np.minimum.accumulate(ge_l[::-1])[::-1]
        ge_r = np.maximum.accumulate(ge_r)
        # findInterval + 1
        gg_1 = np.searchsorted(ge_r + cisDist + 1, sn_l, side="right") + 1
        gg_2 = np.searchsorted(ge_l - cisDist, sn_r, side="right")
    else:
        gg_1 = gg_2 = None

    # ---- orthonormalisation of SNP slices --------------------------
    def orthonormalize_snps(cursnps, ss):
        div = None
        for p in range(len(cursnps)):
            mat = cursnps[p]
            if correction is not None:
                mat = mat @ correction
            rowsq1 = np.sum(mat ** 2, axis=1)
            mat = mat - (mat @ cvrt_mat.T) @ cvrt_mat
            for w in range(p):
                mat = mat - np.sum(mat * cursnps[w], axis=1)[:, None] * cursnps[w]
            rowsq2 = np.sum(mat ** 2, axis=1)
            delete_rows = rowsq2 <= rowsq1 * _EPS
            mat = mat.copy()
            mat[delete_rows, :] = 0.0
            div = np.sqrt(rowsq2)
            div = div.copy()
            div[delete_rows] = 1.0
            cursnps[p] = mat / div[:, None]
        model_snps_std[ss] = div
        return cursnps

    # ---- main double loop ------------------------------------------
    status("Performing eQTL analysis")
    n_tests_all = 0
    n_tests_cis = 0
    n_eqtls_tra = 0
    n_eqtls_cis = 0

    snps_offset = 0
    for ss in range(1, snps.n_slices() + 1):
        cursnps = None
        nrcs = snps.get_n_rows_in_slice(ss)
        if pvOutputThreshold > 0:
            loopset = range(1, gene.n_slices() + 1)
        else:
            a, b = int(gg_1[ss - 1]), int(gg_2[ss - 1])
            loopset = range(a, b + 1) if a <= b else range(0)

        for gg in loopset:
            gene_offset = gene_offsets[gg - 1]
            curgene = gene.get_slice(gg)
            nrcg = curgene.shape[0]
            if nrcg == 0:
                continue

            statistic = None
            astatistic = None
            mat = None
            select_cis_raw = None
            xx = None

            do_cis_here = (
                pvOutputThreshold_cis > 0
                and gg >= gg_1[ss - 1]
                and gg <= gg_2[ss - 1]
            )
            if do_cis_here:
                if cursnps is None:
                    cursnps = orthonormalize_snps(
                        snps_process(snps.get_slice(ss)), ss
                    )
                mat = [curgene @ cs.T for cs in cursnps]
                statistic = statistic_fun(mat)
                astatistic = afun(statistic)

                # cis index selection
                gl = geneloc[gg - 1]
                sl_loc = snpsloc[ss - 1]
                sn_l_g = np.searchsorted(sl_loc, gl[:, 0] - cisDist - 1, side="right")
                sn_r_g = np.searchsorted(sl_loc, gl[:, 1] + cisDist, side="right")
                nrow_stat = statistic.shape[0]
                xx_parts = []
                for x in np.where(sn_r_g > sn_l_g)[0]:
                    cols = np.arange(sn_l_g[x], sn_r_g[x])
                    # R column-major linear index: col*nrow + row
                    xx_parts.append(cols * nrow_stat + x)
                xx = (np.concatenate(xx_parts) if xx_parts
                      else np.array([], dtype=int))
                astat_flat = astatistic.ravel(order="F")
                select_cis_raw = xx[astat_flat[xx] >= thresh_cis]
                sel_rows, sel_cols = _array_ind(select_cis_raw, statistic.shape)
                n_tests_cis += len(xx)
                n_eqtls_cis += len(select_cis_raw)

                if do_hist:
                    hist_cis.update(astat_flat[xx])
                if min_pv_by_genesnp:
                    temp = np.zeros(astatistic.size)
                    temp[xx] = astat_flat[xx]
                    minpv_cis.update(ss, gg, temp.reshape(astatistic.shape, order="F"))

                beta = None
                if model["betafun"] is not None:
                    stat_flat = mat[-1].ravel(order="F")
                    beta = betafun(stat_flat[select_cis_raw], ss, gg,
                                   sel_rows, sel_cols)
                stat_vals = statistic.ravel(order="F")[select_cis_raw]
                saver_cis.update(snps_offset + sel_cols,
                                 gene_offset + sel_rows, stat_vals, beta)

            if pvOutputThreshold > 0:
                if statistic is None:
                    if cursnps is None:
                        cursnps = orthonormalize_snps(
                            snps_process(snps.get_slice(ss)), ss
                        )
                    mat = [curgene @ cs.T for cs in cursnps]
                    statistic = statistic_fun(mat)
                    astatistic = afun(statistic)
                if do_hist:
                    hist_all.update(astatistic)
                astat_work = astatistic
                if select_cis_raw is not None:
                    # remove cis pairs from the trans pass
                    astat_work = astatistic.copy()
                    flat = astat_work.ravel(order="F")
                    flat[xx] = -1.0
                    astat_work = flat.reshape(astatistic.shape, order="F")
                flat_work = astat_work.ravel(order="F")
                select_tra_raw = np.where(flat_work >= thresh)[0]
                sel_rows, sel_cols = _array_ind(select_tra_raw, statistic.shape)
                n_eqtls_tra += len(select_tra_raw)
                n_tests_all += statistic.size

                beta = None
                if model["betafun"] is not None:
                    stat_flat = mat[-1].ravel(order="F")
                    beta = betafun(stat_flat[select_tra_raw], ss, gg,
                                   sel_rows, sel_cols)
                stat_vals = statistic.ravel(order="F")[select_tra_raw]
                saver_tra.update(snps_offset + sel_cols,
                                 gene_offset + sel_rows, stat_vals, beta)
                if min_pv_by_genesnp:
                    minpv_tra.update(ss, gg, astat_work)

        snps_offset += nrcs

    # ---- assemble result -------------------------------------------
    rez = MatrixEQTLResult()
    rez.time_in_sec = time.time() - start_time
    rez.param = params

    if pvOutputThreshold_cis > 0:
        rez._cis = saver_cis.get_results(gene, snps, n_tests_cis)
        rez.cis_ntests = n_tests_cis
        rez.cis_neqtls = n_eqtls_cis
        if do_hist:
            rez.cis_hist_bins, rez.cis_hist_counts = hist_cis.get_results()
        if min_pv_by_genesnp:
            rez.cis_min_pv_snps, rez.cis_min_pv_gene = minpv_cis.get_results(
                snps, gene, lambda x: pvfun(testfun(x))
            )

    if pvOutputThreshold > 0:
        rez.all_ntests = n_tests_all
        rez.all_neqtls = n_eqtls_tra + n_eqtls_cis
        if pvOutputThreshold_cis > 0:
            rez._trans = saver_tra.get_results(gene, snps, n_tests_all - n_tests_cis)
            rez.trans_ntests = n_tests_all - n_tests_cis
            rez.trans_neqtls = n_eqtls_tra
        else:
            rez._all = saver_tra.get_results(gene, snps, n_tests_all)
        if do_hist:
            rez.all_hist_bins, rez.all_hist_counts = hist_all.get_results()
            if pvOutputThreshold_cis > 0:
                rez.trans_hist_bins = rez.all_hist_bins
                rez.trans_hist_counts = rez.all_hist_counts - rez.cis_hist_counts
        if min_pv_by_genesnp:
            if pvOutputThreshold_cis > 0:
                rez.trans_min_pv_snps, rez.trans_min_pv_gene = (
                    minpv_tra.get_results(snps, gene, lambda x: pvfun(testfun(x)))
                )
            else:
                rez.all_min_pv_snps, rez.all_min_pv_gene = (
                    minpv_tra.get_results(snps, gene, lambda x: pvfun(testfun(x)))
                )

    # ---- optional TSV output ---------------------------------------
    if output_file_name and pvOutputThreshold > 0:
        df = rez._trans if rez._trans is not None else rez._all
        if df is not None:
            df.to_csv(output_file_name, sep="\t", index=False)
    if output_file_name_cis and pvOutputThreshold_cis > 0 and rez._cis is not None:
        rez._cis.to_csv(output_file_name_cis, sep="\t", index=False)

    status("")
    return rez


def Matrix_eQTL_engine(  # noqa: N802 -- R-compatible name
    snps: SlicedData,
    gene: SlicedData,
    cvrt: Optional[SlicedData] = None,
    output_file_name: Optional[str] = None,
    pvOutputThreshold: float = 1e-5,
    useModel: int = modelLINEAR,
    errorCovariance=None,
    verbose: bool = False,
    pvalue_hist=False,
    min_pv_by_genesnp: bool = False,
    noFDRsaveMemory: bool = False,
) -> MatrixEQTLResult:
    """Thin wrapper around :func:`Matrix_eQTL_main` (R ``Matrix_eQTL_engine``).

    Runs a trans-only (all-pairs) eQTL analysis -- no cis/trans split.
    """
    return Matrix_eQTL_main(
        snps=snps,
        gene=gene,
        cvrt=cvrt,
        output_file_name=output_file_name,
        pvOutputThreshold=pvOutputThreshold,
        useModel=useModel,
        errorCovariance=errorCovariance,
        verbose=verbose,
        pvalue_hist=pvalue_hist,
        min_pv_by_genesnp=min_pv_by_genesnp,
        noFDRsaveMemory=noFDRsaveMemory,
    )
