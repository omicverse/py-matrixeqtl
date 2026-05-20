# py-matrixeqtl

Pure-Python port of the R/CRAN package **[MatrixEQTL](https://cran.r-project.org/package=MatrixEQTL)** —
Andrey A. Shabalin's ultra-fast eQTL mapping engine
(Shabalin, *Bioinformatics* 2012, 28(10):1353–1358).

`pymatrixeqtl` is a standalone, dependency-light (numpy / scipy / pandas
only) reimplementation of MatrixEQTL's complete computational core. It
reproduces the upstream results **bit-for-bit** — there is no R involved.

| | |
|---|---|
| PyPI / import name | `pymatrixeqtl` |
| License | LGPL-3 (same as upstream MatrixEQTL) |
| Upstream | CRAN MatrixEQTL 2.3 |
| Dependencies | numpy, scipy, pandas |

## What it does

MatrixEQTL tests every SNP–gene pair for association extremely fast. The
trick — faithfully replicated here — is to residualise the expression and
genotype matrices on the covariates **once**, scale each row to unit norm,
and then obtain every test statistic as a single entry of one big matrix
product (a correlation). The t / F statistic and p-value follow
analytically.

* **Three models**
  * `modelLINEAR` — additive linear model `expr ~ SNP + covariates`
    (reports slope `beta`, t-statistic, p-value).
  * `modelANOVA` — genotype as a categorical factor (F-test).
  * `modelLINEAR_CROSS` — linear model with a SNP × last-covariate
    interaction term.
* **cis / trans split** — with SNP- and gene-position tables, associations
  are split into *cis* (within `cisDist` bp, default 1e6) and *trans*,
  each with its own p-value threshold.
* **Benjamini-Hochberg FDR** — computed exactly as MatrixEQTL does, over
  *all* tested pairs (not only those passing the threshold).
* **Error-covariance whitening**, the `min.pv.by.genesnp` tables and
  p-value histograms / Q-Q profiles — all ported faithfully.

## Install

```bash
pip install pymatrixeqtl            # once published
# or, from a checkout:
pip install -e .
```

## Quick start

```python
import pymatrixeqtl as me

# MatrixEQTL ships an example dataset; it is bundled inside this package.
ex = me.load_example()

# cis / trans eQTL analysis, additive linear model
res = me.Matrix_eQTL_main(
    snps=ex["snps"], gene=ex["gene"], cvrt=ex["cvrt"],
    pvOutputThreshold=1e-2,            # trans threshold
    pvOutputThreshold_cis=1e-1,        # cis threshold
    snpspos=ex["snpspos"], genepos=ex["genepos"],
    useModel=me.modelLINEAR, cisDist=1e6,
)

res.cis     # tidy DataFrame: snps, gene, beta, statistic, pvalue, FDR
res.trans   # tidy DataFrame
print(res.summary())
```

Or use the one-call high-level wrapper, which accepts file paths,
NumPy arrays or pandas DataFrames:

```python
res = me.eqtl(
    "SNP.txt", "GE.txt", "Covariates.txt",
    model=me.modelANOVA, pv_threshold=1e-5,
    snpspos="snpsloc.txt", genepos="geneloc.txt",
    pv_threshold_cis=1e-3, cis_dist=1e6,
)
```

## The `SlicedData` container

MatrixEQTL's chunked matrix container is ported as
`pymatrixeqtl.SlicedData`, a NumPy-backed class with the row-slice
chunking preserved for big-matrix memory parity.

| R method | Python method |
|---|---|
| `$new(mat)` / `CreateFromMatrix` | `SlicedData(mat)` / `create_from_matrix` |
| `LoadFile` | `load_file` |
| `nRows()` / `nCols()` / `nSlices()` | `n_rows()` / `n_cols()` / `n_slices()` |
| `RowStandardizeCentered` | `row_standardize_centered` |
| `ColumnSubsample` | `column_subsample` |
| `RowReorder` | `row_reorder` |
| `ResliceCombined` | `reslice_combined` |
| `CombineInOneSlice` | `combine_in_one_slice` |
| `FindRow` | `find_row` |
| `SetNanRowMean` | `set_nan_row_mean` |

The original R names are also available as aliases (`SlicedData.LoadFile`,
`SlicedData.nRows`, …).

## Public API

`SlicedData`, `Matrix_eQTL_main`, `Matrix_eQTL_engine`, `MatrixEQTLResult`,
`modelLINEAR`, `modelANOVA`, `modelLINEAR_CROSS`, `MODEL_NAMES`, `eqtl`,
`load_example`, `EXAMPLE_DIR`, `plot_matrix_eqtl`.

## Accuracy vs R

`pymatrixeqtl` is verified against CRAN MatrixEQTL 2.3 on MatrixEQTL's own
bundled example dataset for all three models, both cis and trans:

| quantity | Pearson r vs R | max \|diff\| |
|---|---|---|
| beta (slope) | 1.0000000 | < 2e-15 |
| t / F statistic | 1.0000000 | < 1.1e-11 |
| p-value | 1.0000000 | < 2e-14 |
| FDR | 1.0000000 | < 3e-14 |

cis / trans split counts match exactly. Differences are pure
floating-point round-off — this is the same linear algebra.

Run the parity tests yourself:

```bash
pytest tests/ -q
```

`tests/test_r_parity.py` drives R MatrixEQTL via the bundled
`tests/r_reference_driver.R` and skips gracefully if R is unavailable;
`tests/test_smoke.py` is pure Python.

## Benchmark

```bash
python examples/benchmark.py --big
```

On a 5000 SNP × 2000 gene × 200 sample synthetic dataset the Python
engine tests all 10,000,000 pairs in ~0.3 s.

See `examples/compare_R_vs_Python.ipynb` for a full executed R-vs-Python
timing / accuracy comparison with plots.

## Citation

If you use this software, please cite the original MatrixEQTL paper:

> Shabalin, A. A. (2012). Matrix eQTL: ultra fast eQTL analysis via large
> matrix operations. *Bioinformatics*, 28(10), 1353–1358.

## License

LGPL-3, matching upstream MatrixEQTL. See `LICENSE`.
