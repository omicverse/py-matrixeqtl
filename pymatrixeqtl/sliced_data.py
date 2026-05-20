"""SlicedData -- MatrixEQTL's chunked matrix container.

This is a faithful Python port of MatrixEQTL's ``SlicedData`` R5 reference
class (``R/SlicedData.r`` in the upstream package).  ``SlicedData`` stores a
genotype / expression / covariate matrix split into horizontal *slices*
(blocks of rows) so very large matrices can be processed block-by-block
without ever materialising the whole thing.

The container is backed by NumPy.  Each slice is a 2-D ``float64`` array;
missing values are stored as ``np.nan``.  Row names are kept per-slice (a
list of arrays), column (sample) names are global.

R API -> Python API mapping
---------------------------
======================  ==================================
R method                Python method
======================  ==================================
``$new(mat)``           ``SlicedData(mat)``
``CreateFromMatrix``    :meth:`SlicedData.create_from_matrix`
``LoadFile``            :meth:`SlicedData.load_file`
``SaveFile``            :meth:`SlicedData.save_file`
``Clone``               :meth:`SlicedData.clone`
``Clear``               :meth:`SlicedData.clear`
``nRows()``             :meth:`SlicedData.n_rows`
``nCols()``             :meth:`SlicedData.n_cols`
``nSlices()``           :meth:`SlicedData.n_slices`
``getSlice(sl)``        :meth:`SlicedData.get_slice`  (1-based, R-style)
``setSlice(sl, v)``     :meth:`SlicedData.set_slice`
``GetNRowsInSlice``     :meth:`SlicedData.get_n_rows_in_slice`
``GetAllRowNames``      :meth:`SlicedData.get_all_row_names`
``RowStandardizeCentered`` :meth:`SlicedData.row_standardize_centered`
``ColumnSubsample``     :meth:`SlicedData.column_subsample`
``RowReorder``          :meth:`SlicedData.row_reorder`
``ResliceCombined``     :meth:`SlicedData.reslice_combined`
``CombineInOneSlice``   :meth:`SlicedData.combine_in_one_slice`
``FindRow``             :meth:`SlicedData.find_row`
``SetNanRowMean``       :meth:`SlicedData.set_nan_row_mean`
``RowMatrixMultiply``   :meth:`SlicedData.row_matrix_multiply`
``RowRemoveZeroEps``    :meth:`SlicedData.row_remove_zero_eps`
======================  ==================================

For convenience the R names are also exposed as method aliases, and the
fields ``nSlices``/``columnNames``/``fileSliceSize`` carry their R names.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["SlicedData"]

_EPS = np.finfo(np.float64).eps


class SlicedData:
    """Chunked, NumPy-backed matrix container.

    Parameters
    ----------
    mat :
        Optional matrix to load immediately.  Accepts a 2-D ``numpy``
        array, a :class:`pandas.DataFrame` (its index/columns become row /
        column names) or anything array-like.
    row_names, col_names :
        Optional explicit names, used when ``mat`` carries none.
    file_slice_size :
        Rows per slice when (re)slicing.  Mirrors R's ``fileSliceSize``
        (default 1000).
    """

    def __init__(
        self,
        mat=None,
        row_names: Optional[Sequence] = None,
        col_names: Optional[Sequence] = None,
        file_slice_size: int = 1000,
    ) -> None:
        self._slices: List[np.ndarray] = []
        self.rowNameSlices: List[np.ndarray] = []
        self.columnNames: np.ndarray = np.array([], dtype=object)
        # file-reading parameters (R fields)
        self.fileSliceSize = int(file_slice_size)
        self.fileDelimiter = "\t"
        self.fileSkipColumns = 1
        self.fileSkipRows = 1
        self.fileOmitCharacters = "NA"
        if mat is not None:
            self.create_from_matrix(mat, row_names=row_names, col_names=col_names)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def create_from_matrix(self, mat, row_names=None, col_names=None) -> "SlicedData":
        """Load a whole matrix as a single slice (R ``CreateFromMatrix``)."""
        if isinstance(mat, pd.DataFrame):
            if row_names is None:
                row_names = mat.index.to_numpy()
            if col_names is None:
                col_names = mat.columns.to_numpy()
            arr = mat.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(mat, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        nr, nc = arr.shape
        if row_names is None:
            row_names = np.array([f"Row_{i + 1}" for i in range(nr)], dtype=object)
        else:
            row_names = np.asarray(list(row_names), dtype=object)
        if col_names is None:
            col_names = np.array([f"Col_{i + 1}" for i in range(nc)], dtype=object)
        else:
            col_names = np.asarray(list(col_names), dtype=object)
        if len(row_names) != nr:
            raise ValueError("row_names length does not match matrix rows")
        if len(col_names) != nc:
            raise ValueError("col_names length does not match matrix columns")
        self._slices = [arr.copy()]
        self.rowNameSlices = [row_names]
        self.columnNames = col_names
        return self

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, file_slice_size: int = 1000) -> "SlicedData":
        """Build a :class:`SlicedData` from a pandas DataFrame."""
        return cls(df, file_slice_size=file_slice_size)

    def load_file(
        self,
        filename: str,
        skip_rows: Optional[int] = None,
        skip_columns: Optional[int] = None,
        slice_size: Optional[int] = None,
        omit_characters: Optional[str] = None,
        delimiter: Optional[str] = None,
        row_names_column: int = 1,
    ) -> "SlicedData":
        """Read a delimited text file with row + column names (R ``LoadFile``).

        The file layout matches MatrixEQTL's example data: the first
        ``fileSkipRows`` line(s) are a header, the first ``fileSkipColumns``
        column(s) hold identifiers, and ``row_names_column`` (1-based) picks
        which of those is used as the row name.
        """
        if skip_rows is not None:
            self.fileSkipRows = int(skip_rows)
        if skip_columns is not None:
            self.fileSkipColumns = int(skip_columns)
        if omit_characters is not None:
            self.fileOmitCharacters = omit_characters
        if slice_size is not None:
            self.fileSliceSize = int(slice_size)
        if delimiter is not None:
            self.fileDelimiter = delimiter
        if self.fileSkipColumns != 0 and not (1 <= row_names_column <= self.fileSkipColumns):
            raise ValueError("row_names_column out of range vs fileSkipColumns")

        header = self.fileSkipRows if self.fileSkipRows > 0 else None
        na_vals = [self.fileOmitCharacters] if self.fileOmitCharacters else None
        df = pd.read_csv(
            filename,
            sep=self.fileDelimiter,
            header=(self.fileSkipRows - 1) if self.fileSkipRows > 0 else None,
            na_values=na_vals,
            engine="python",
        )
        ncols_total = df.shape[1]
        if self.fileSkipColumns > 0:
            row_names = df.iloc[:, row_names_column - 1].astype(str).to_numpy(dtype=object)
            data = df.iloc[:, self.fileSkipColumns:].to_numpy(dtype=np.float64)
            if self.fileSkipRows > 0:
                col_names = np.asarray(
                    list(df.columns[self.fileSkipColumns:]), dtype=object
                )
            else:
                col_names = np.array(
                    [f"Col_{i + 1}" for i in range(data.shape[1])], dtype=object
                )
        else:
            data = df.to_numpy(dtype=np.float64)
            row_names = np.array(
                [f"Row_{i + 1}" for i in range(data.shape[0])], dtype=object
            )
            col_names = (
                np.asarray(list(df.columns), dtype=object)
                if self.fileSkipRows > 0
                else np.array([f"Col_{i + 1}" for i in range(data.shape[1])], dtype=object)
            )
        del header, ncols_total
        self.create_from_matrix(data, row_names=row_names, col_names=col_names)
        self.reslice_combined()
        return self

    def save_file(self, filename: str) -> None:
        """Write the matrix to a tab-delimited file (R ``SaveFile``)."""
        self.as_dataframe().to_csv(filename, sep="\t")

    # ------------------------------------------------------------------
    # dimensions
    # ------------------------------------------------------------------
    def n_slices(self) -> int:
        """Number of slices."""
        return len(self._slices)

    @property
    def nSlices(self) -> int:  # noqa: N802 - R-compatible name
        return len(self._slices)

    def n_cols(self) -> int:
        """Number of columns (samples)."""
        return 0 if not self._slices else self._slices[0].shape[1]

    def n_rows(self) -> int:
        """Total number of rows across all slices."""
        return int(sum(s.shape[0] for s in self._slices))

    def get_n_rows_in_slice(self, sl: int) -> int:
        """Number of rows in slice ``sl`` (1-based)."""
        return self.rowNameSlices[sl - 1].shape[0]

    # R aliases
    nRows = n_rows
    nCols = n_cols
    nSlicesMethod = n_slices
    GetNRowsInSlice = get_n_rows_in_slice

    @property
    def shape(self):
        return (self.n_rows(), self.n_cols())

    def is_combined(self) -> bool:
        """True if stored in a single slice (R ``IsCombined``)."""
        return self.n_slices() <= 1

    IsCombined = is_combined

    # ------------------------------------------------------------------
    # slice access  (1-based indices, R-style)
    # ------------------------------------------------------------------
    def get_slice(self, sl: int) -> np.ndarray:
        """Return slice ``sl`` (1-based) as a float64 array."""
        return self._slices[sl - 1]

    def set_slice(self, sl: int, value) -> None:
        """Replace slice ``sl`` (1-based) with ``value``."""
        arr = np.asarray(value, dtype=np.float64)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        while len(self._slices) < sl:
            self._slices.append(np.empty((0, 0)))
        self._slices[sl - 1] = arr

    getSlice = get_slice
    setSlice = set_slice
    getSliceRaw = get_slice
    setSliceRaw = set_slice

    def get_all_row_names(self) -> np.ndarray:
        """Concatenated row names across all slices (R ``GetAllRowNames``)."""
        if not self.rowNameSlices:
            return np.array([], dtype=object)
        return np.concatenate(self.rowNameSlices)

    GetAllRowNames = get_all_row_names

    @property
    def row_names(self) -> np.ndarray:
        return self.get_all_row_names()

    @property
    def col_names(self) -> np.ndarray:
        return self.columnNames

    def clear(self) -> "SlicedData":
        """Drop all data (R ``Clear``)."""
        self._slices = []
        self.rowNameSlices = []
        self.columnNames = np.array([], dtype=object)
        return self

    Clear = clear

    def clone(self) -> "SlicedData":
        """Deep copy (R ``Clone``)."""
        new = SlicedData(file_slice_size=self.fileSliceSize)
        new._slices = [s.copy() for s in self._slices]
        new.rowNameSlices = [r.copy() for r in self.rowNameSlices]
        new.columnNames = self.columnNames.copy()
        new.fileDelimiter = self.fileDelimiter
        new.fileSkipColumns = self.fileSkipColumns
        new.fileSkipRows = self.fileSkipRows
        new.fileOmitCharacters = self.fileOmitCharacters
        return new

    Clone = clone
    copy = clone

    # ------------------------------------------------------------------
    # reshaping
    # ------------------------------------------------------------------
    def combine_in_one_slice(self) -> "SlicedData":
        """Merge all slices into a single slice (R ``CombineInOneSlice``)."""
        if self.n_slices() <= 1:
            return self
        data = np.vstack(self._slices)
        names = self.get_all_row_names()
        self._slices = [data]
        self.rowNameSlices = [names]
        return self

    CombineInOneSlice = combine_in_one_slice

    def reslice_combined(self, slice_size: int = -1) -> "SlicedData":
        """Re-split a combined matrix into slices of ``fileSliceSize`` rows.

        Mirrors R's ``ResliceCombined``.  Requires the data to be combined.
        """
        if slice_size > 0:
            self.fileSliceSize = int(slice_size)
        if self.fileSliceSize <= 0:
            self.fileSliceSize = 1000
        if not self.is_combined():
            raise ValueError(
                "Reslice of a sliced matrix is not supported. "
                "Use combine_in_one_slice first."
            )
        nrows = self.n_rows()
        if nrows == 0:
            return self
        data = self._slices[0]
        names = self.rowNameSlices[0]
        new_slices = []
        new_names = []
        n_new = (nrows + self.fileSliceSize - 1) // self.fileSliceSize
        for i in range(n_new):
            fr = i * self.fileSliceSize
            to = min((i + 1) * self.fileSliceSize, nrows)
            new_slices.append(data[fr:to, :].copy())
            new_names.append(names[fr:to].copy())
        self._slices = new_slices
        self.rowNameSlices = new_names
        return self

    ResliceCombined = reslice_combined

    def row_reorder(self, ordr) -> "SlicedData":
        """Reorder rows by index/boolean mask (R ``RowReorder``).

        ``ordr`` may be a 0-based integer permutation or a boolean mask.
        """
        ordr = np.asarray(ordr)
        if ordr.dtype == bool:
            if ordr.shape[0] != self.n_rows():
                raise ValueError('Parameter "ordr" has wrong length')
            ordr = np.where(ordr)[0]
        else:
            ordr = ordr.astype(int)
        if ordr.shape[0] == self.n_rows() and np.array_equal(
            ordr, np.arange(self.n_rows())
        ):
            return self
        self.combine_in_one_slice()
        data = self._slices[0][ordr, :]
        names = self.rowNameSlices[0][ordr]
        self._slices = [data]
        self.rowNameSlices = [names]
        self.reslice_combined()
        return self

    RowReorder = row_reorder
    RowReorderSimple = row_reorder

    def column_subsample(self, subset) -> "SlicedData":
        """Keep only the given columns (R ``ColumnSubsample``).

        ``subset`` is a 0-based integer index array or boolean mask.
        """
        subset = np.asarray(subset)
        for i in range(self.n_slices()):
            self._slices[i] = self._slices[i][:, subset]
        self.columnNames = self.columnNames[subset]
        return self

    ColumnSubsample = column_subsample

    # ------------------------------------------------------------------
    # numerical transforms
    # ------------------------------------------------------------------
    def row_standardize_centered(self) -> "SlicedData":
        """Scale each row to unit L2 norm (R ``RowStandardizeCentered``)."""
        for i in range(self.n_slices()):
            slice_ = self._slices[i]
            div = np.sqrt(np.sum(slice_ ** 2, axis=1))
            div[div == 0] = 1.0
            self._slices[i] = slice_ / div[:, None]
        return self

    RowStandardizeCentered = row_standardize_centered

    def set_nan_row_mean(self) -> "SlicedData":
        """Impute NaNs with the row mean (R ``SetNanRowMean``)."""
        if self.n_cols() == 0:
            return self
        for i in range(self.n_slices()):
            slice_ = self._slices[i]
            if np.isnan(slice_).any():
                slice_ = slice_.copy()
                with np.errstate(invalid="ignore"):
                    rowmean = np.nanmean(slice_, axis=1)
                rowmean[np.isnan(rowmean)] = 0.0
                inds = np.where(np.isnan(slice_).any(axis=1))[0]
                for j in inds:
                    nanmask = np.isnan(slice_[j, :])
                    slice_[j, nanmask] = rowmean[j]
                self._slices[i] = slice_
        return self

    SetNanRowMean = set_nan_row_mean

    def row_matrix_multiply(self, multiplier) -> "SlicedData":
        """Right-multiply every slice by ``multiplier`` (R ``RowMatrixMultiply``)."""
        multiplier = np.asarray(multiplier, dtype=np.float64)
        for i in range(self.n_slices()):
            self._slices[i] = self._slices[i] @ multiplier
        return self

    RowMatrixMultiply = row_matrix_multiply

    def row_remove_zero_eps(self) -> "SlicedData":
        """Drop near-zero rows (R ``RowRemoveZeroEps``)."""
        ncol = self.n_cols()
        for i in range(self.n_slices()):
            slice_ = self._slices[i]
            amean = np.mean(np.abs(slice_), axis=1)
            keep = amean >= _EPS * ncol
            if not keep.all():
                self._slices[i] = slice_[keep, :]
                self.rowNameSlices[i] = self.rowNameSlices[i][keep]
        return self

    RowRemoveZeroEps = row_remove_zero_eps

    # ------------------------------------------------------------------
    # lookup / export
    # ------------------------------------------------------------------
    def find_row(self, rowname):
        """Locate a row by name (R ``FindRow``).

        Returns ``dict(slice=<1-based>, item=<1-based>, row=<DataFrame>)``
        or ``None`` if not found.
        """
        for sl in range(self.n_slices()):
            names = self.rowNameSlices[sl]
            matches = np.where(names == rowname)[0]
            if matches.size > 0:
                item = int(matches[0])
                row = self._slices[sl][item:item + 1, :]
                return {
                    "slice": sl + 1,
                    "item": item + 1,
                    "row": pd.DataFrame(row, index=[rowname], columns=self.columnNames),
                }
        return None

    FindRow = find_row

    def as_matrix(self) -> np.ndarray:
        """Return the whole matrix as one NumPy array (R ``as.matrix``)."""
        if not self._slices:
            return np.empty((0, 0))
        return np.vstack(self._slices)

    as_array = as_matrix

    def as_dataframe(self) -> pd.DataFrame:
        """Return the whole matrix as a labelled :class:`pandas.DataFrame`."""
        return pd.DataFrame(
            self.as_matrix(),
            index=self.get_all_row_names(),
            columns=self.columnNames,
        )

    # ------------------------------------------------------------------
    # dunder helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_rows()

    def __repr__(self) -> str:
        return (
            f"SlicedData({self.n_rows()} rows x {self.n_cols()} cols, "
            f"{self.n_slices()} slice(s))"
        )

    def __getitem__(self, key):
        """Numpy-style slicing over the combined matrix."""
        return self.as_dataframe().iloc[key]
