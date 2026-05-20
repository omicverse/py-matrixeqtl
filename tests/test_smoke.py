"""Pure-Python smoke tests for :mod:`pymatrixeqtl` (no R required).

These exercise the public API end-to-end on MatrixEQTL's bundled example
dataset and on small synthetic matrices, checking shapes, column names,
the cis/trans split, FDR monotonicity, and the three models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pymatrixeqtl as me
from pymatrixeqtl import SlicedData


# ----------------------------------------------------------------------
# SlicedData
# ----------------------------------------------------------------------
def test_sliced_data_from_matrix():
    arr = np.arange(12, dtype=float).reshape(3, 4)
    sd = SlicedData(arr)
    assert sd.n_rows() == 3
    assert sd.n_cols() == 4
    assert sd.n_slices() == 1
    assert list(sd.row_names) == ["Row_1", "Row_2", "Row_3"]
    np.testing.assert_array_equal(sd.as_matrix(), arr)


def test_sliced_data_from_dataframe():
    df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["g1", "g2"],
        columns=["s1", "s2"],
    )
    sd = SlicedData(df)
    assert list(sd.row_names) == ["g1", "g2"]
    assert list(sd.col_names) == ["s1", "s2"]


def test_sliced_data_load_file():
    ex = me.load_example()
    sd = SlicedData()
    sd.load_file(ex["paths"]["snps"])
    assert sd.shape == (15, 16)
    assert sd.row_names[0] == "Snp_01"
    assert sd.col_names[0] == "Sam_01"


def test_sliced_data_reslice_and_combine():
    arr = np.random.RandomState(0).randn(25, 6)
    sd = SlicedData(arr)
    sd.reslice_combined(slice_size=10)
    assert sd.n_slices() == 3
    assert sd.get_n_rows_in_slice(1) == 10
    assert sd.get_n_rows_in_slice(3) == 5
    sd.combine_in_one_slice()
    assert sd.n_slices() == 1
    np.testing.assert_allclose(sd.as_matrix(), arr)


def test_sliced_data_row_reorder():
    arr = np.arange(15, dtype=float).reshape(5, 3)
    sd = SlicedData(arr)
    sd.row_reorder(np.array([4, 3, 2, 1, 0]))
    np.testing.assert_array_equal(sd.as_matrix(), arr[::-1])
    assert list(sd.row_names) == ["Row_5", "Row_4", "Row_3", "Row_2", "Row_1"]


def test_sliced_data_column_subsample():
    arr = np.arange(12, dtype=float).reshape(3, 4)
    sd = SlicedData(arr)
    sd.column_subsample(np.array([0, 2]))
    assert sd.n_cols() == 2
    np.testing.assert_array_equal(sd.as_matrix(), arr[:, [0, 2]])


def test_sliced_data_row_standardize_centered():
    arr = np.array([[3.0, 4.0], [0.0, 0.0]])
    sd = SlicedData(arr)
    sd.row_standardize_centered()
    np.testing.assert_allclose(np.sum(sd.as_matrix()[0] ** 2), 1.0)
    # zero row stays zero
    np.testing.assert_array_equal(sd.as_matrix()[1], [0.0, 0.0])


def test_sliced_data_set_nan_row_mean():
    arr = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])
    sd = SlicedData(arr)
    sd.set_nan_row_mean()
    assert sd.as_matrix()[0, 1] == pytest.approx(2.0)
    np.testing.assert_array_equal(sd.as_matrix()[1], [0.0, 0.0, 0.0])


def test_sliced_data_find_row():
    sd = SlicedData(np.eye(3), row_names=["a", "b", "c"])
    hit = sd.find_row("b")
    assert hit is not None
    assert hit["item"] == 2
    np.testing.assert_array_equal(hit["row"].to_numpy().ravel(), [0, 1, 0])
    assert sd.find_row("zzz") is None


def test_sliced_data_clone_independent():
    sd = SlicedData(np.ones((2, 2)))
    cl = sd.clone()
    cl.get_slice(1)[0, 0] = 99.0
    assert sd.get_slice(1)[0, 0] == 1.0


# ----------------------------------------------------------------------
# example data loader
# ----------------------------------------------------------------------
def test_load_example():
    ex = me.load_example()
    assert ex["snps"].shape == (15, 16)
    assert ex["gene"].shape == (10, 16)
    assert ex["cvrt"].shape == (2, 16)
    assert list(ex["snpspos"].columns) == ["snpid", "chr", "pos"]
    assert list(ex["genepos"].columns) == ["geneid", "chr", "left", "right"]


# ----------------------------------------------------------------------
# engine: trans-only, all three models
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "model", [me.modelLINEAR, me.modelANOVA, me.modelLINEAR_CROSS]
)
def test_engine_all_models(model):
    ex = me.load_example()
    res = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1.0, useModel=model,
    )
    df = res.all
    assert len(df) == 15 * 10
    for col in ("snps", "gene", "statistic", "pvalue", "FDR"):
        assert col in df.columns
    # p-values in (0, 1]
    assert (df["pvalue"] > 0).all() and (df["pvalue"] <= 1).all()
    # FDR is non-decreasing when sorted by descending |statistic|
    assert np.all(np.diff(df["FDR"].to_numpy()) >= -1e-9)
    assert res.all_ntests == 150


def test_engine_linear_has_beta():
    ex = me.load_example()
    res = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1.0, useModel=me.modelLINEAR,
    )
    assert "beta" in res.all.columns
    assert res.all["beta"].notna().all()


def test_engine_anova_no_beta():
    ex = me.load_example()
    res = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1.0, useModel=me.modelANOVA,
    )
    # ANOVA does not report a slope
    assert "beta" not in res.all.columns


def test_engine_threshold_filters():
    ex = me.load_example()
    full = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1.0, useModel=me.modelLINEAR,
    )
    strict = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1e-3, useModel=me.modelLINEAR,
    )
    assert len(strict.all) < len(full.all)
    assert (strict.all["pvalue"] <= 1e-3).all()
    # ntests still counts ALL pairs even though only some are reported
    assert strict.all_ntests == 150


# ----------------------------------------------------------------------
# engine: cis / trans split
# ----------------------------------------------------------------------
def test_engine_cis_trans_split():
    ex = me.load_example()
    res = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1.0, pvOutputThreshold_cis=1.0,
        snpspos=ex["snpspos"], genepos=ex["genepos"],
        useModel=me.modelLINEAR, cisDist=1e6,
    )
    assert res.cis_ntests + res.trans_ntests == 150
    assert len(res.cis) == res.cis_ntests
    assert len(res.trans) == res.trans_ntests
    # a cis pair must never also appear as a trans pair
    cis_pairs = set(zip(res.cis["snps"], res.cis["gene"]))
    trans_pairs = set(zip(res.trans["snps"], res.trans["gene"]))
    assert cis_pairs.isdisjoint(trans_pairs)


def test_engine_cis_only():
    ex = me.load_example()
    res = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=0.0, pvOutputThreshold_cis=1.0,
        snpspos=ex["snpspos"], genepos=ex["genepos"],
        useModel=me.modelLINEAR, cisDist=1e6,
    )
    assert len(res.cis) > 0
    assert len(res.trans) == 0


# ----------------------------------------------------------------------
# engine extras
# ----------------------------------------------------------------------
def test_engine_min_pv_by_genesnp():
    ex = me.load_example()
    res = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1.0, useModel=me.modelLINEAR,
        min_pv_by_genesnp=True,
    )
    assert res.all_min_pv_snps.shape == (15,)
    assert res.all_min_pv_gene.shape == (10,)
    # the best p per gene equals the global min over that gene's pairs
    g0 = ex["gene"].row_names[0]
    expected = res.all.loc[res.all["gene"] == g0, "pvalue"].min()
    assert res.all_min_pv_gene[g0] == pytest.approx(expected, rel=1e-6)


def test_engine_pvalue_hist():
    ex = me.load_example()
    res = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1.0, useModel=me.modelLINEAR,
        pvalue_hist=True,
    )
    assert res.all_hist_counts.sum() == 150
    assert len(res.all_hist_bins) == 101


def test_engine_error_covariance_identity_noop():
    """An identity error covariance must not change the result."""
    ex = me.load_example()
    base = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1.0, useModel=me.modelLINEAR,
    )
    with_cov = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1.0, useModel=me.modelLINEAR,
        errorCovariance=np.eye(16),
    )
    m = base.all.merge(with_cov.all, on=["snps", "gene"], suffixes=("_a", "_b"))
    np.testing.assert_allclose(
        m["statistic_a"], m["statistic_b"], atol=1e-8
    )


def test_engine_engine_wrapper():
    ex = me.load_example()
    res = me.Matrix_eQTL_engine(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1.0, useModel=me.modelLINEAR,
    )
    assert len(res.all) == 150


# ----------------------------------------------------------------------
# high-level eqtl()
# ----------------------------------------------------------------------
def test_eqtl_from_paths():
    ex = me.load_example()
    res = me.eqtl(
        ex["paths"]["snps"], ex["paths"]["gene"], ex["paths"]["cvrt"],
        model=me.modelLINEAR, pv_threshold=1.0,
    )
    assert len(res.all) == 150


def test_eqtl_from_dataframes():
    ex = me.load_example()
    snps_df = ex["snps"].as_dataframe()
    gene_df = ex["gene"].as_dataframe()
    cvrt_df = ex["cvrt"].as_dataframe()
    res = me.eqtl(snps_df, gene_df, cvrt_df, pv_threshold=1.0)
    assert len(res.all) == 150


def test_eqtl_cis_with_position_paths():
    ex = me.load_example()
    res = me.eqtl(
        ex["paths"]["snps"], ex["paths"]["gene"], ex["paths"]["cvrt"],
        model=me.modelLINEAR, pv_threshold=1.0, pv_threshold_cis=1.0,
        snpspos=ex["paths"]["snpspos"], genepos=ex["paths"]["genepos"],
        cis_dist=1e6,
    )
    assert len(res.cis) + len(res.trans) == 150


# ----------------------------------------------------------------------
# result object
# ----------------------------------------------------------------------
def test_result_repr_and_summary():
    ex = me.load_example()
    res = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1.0, useModel=me.modelLINEAR,
    )
    s = res.summary()
    assert "modelLINEAR" in s
    assert "MatrixEQTL result" in repr(res)
    assert res.time_in_sec >= 0


# ----------------------------------------------------------------------
# correctness vs an independent OLS implementation
# ----------------------------------------------------------------------
def test_linear_matches_independent_ols():
    """A few linear-model t-stats must match a plain per-pair OLS."""
    ex = me.load_example()
    res = me.Matrix_eQTL_main(
        snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
        pvOutputThreshold=1.0, useModel=me.modelLINEAR,
    )
    snps = ex["snps"].as_dataframe()
    gene = ex["gene"].as_dataframe()
    cvrt = ex["cvrt"].as_dataframe()
    n = snps.shape[1]
    design_base = np.column_stack([np.ones(n), cvrt.to_numpy().T])
    checked = 0
    for _, row in res.all.head(8).iterrows():
        x = snps.loc[row["snps"]].to_numpy()
        y = gene.loc[row["gene"]].to_numpy()
        X = np.column_stack([design_base, x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = n - X.shape[1]
        sigma2 = resid @ resid / dof
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(sigma2 * xtx_inv[-1, -1])
        t = beta[-1] / se
        assert t == pytest.approx(row["statistic"], rel=1e-6, abs=1e-6)
        assert beta[-1] == pytest.approx(row["beta"], rel=1e-6, abs=1e-8)
        checked += 1
    assert checked == 8
