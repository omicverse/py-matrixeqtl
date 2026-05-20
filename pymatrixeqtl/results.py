"""Result container for :mod:`pymatrixeqtl`.

:class:`MatrixEQTLResult` mirrors the named list returned by R's
``Matrix_eQTL_main`` (which carries class ``"MatrixEQTL"``).  It exposes:

* ``.all`` / ``.cis`` / ``.trans`` -- tidy result DataFrames with columns
  ``snps, gene, beta, statistic, pvalue, FDR``.  ``.all`` is populated only
  when there is no cis/trans split; ``.cis`` / ``.trans`` otherwise.
* ``.param`` -- the analysis parameters (R ``$param``).
* ``.time_in_sec`` -- wall-clock runtime (R ``$time.in.sec``).
* ``.cis_ntests`` / ``.trans_ntests`` / ``.all_ntests`` and the matching
  ``*_neqtls`` counts.
* ``min.pv.by.genesnp`` tables when that option is on.
* p-value histogram bins/counts when ``pvalue_hist`` is on.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["MatrixEQTLResult"]

_EMPTY = pd.DataFrame(
    {
        "snps": pd.Series([], dtype=object),
        "gene": pd.Series([], dtype=object),
        "beta": pd.Series([], dtype=float),
        "statistic": pd.Series([], dtype=float),
        "pvalue": pd.Series([], dtype=float),
        "FDR": pd.Series([], dtype=float),
    }
)


class MatrixEQTLResult:
    """Container for the output of a MatrixEQTL run.

    Access the eQTL tables via :attr:`all`, :attr:`cis` and :attr:`trans`.
    """

    def __init__(self) -> None:
        self._all: Optional[pd.DataFrame] = None
        self._cis: Optional[pd.DataFrame] = None
        self._trans: Optional[pd.DataFrame] = None
        self.param: dict = {}
        self.time_in_sec: float = 0.0
        # test / eqtl counts
        self.all_ntests: Optional[int] = None
        self.all_neqtls: Optional[int] = None
        self.cis_ntests: Optional[int] = None
        self.cis_neqtls: Optional[int] = None
        self.trans_ntests: Optional[int] = None
        self.trans_neqtls: Optional[int] = None
        # min p-value tables
        self.all_min_pv_snps: Optional[pd.Series] = None
        self.all_min_pv_gene: Optional[pd.Series] = None
        self.cis_min_pv_snps: Optional[pd.Series] = None
        self.cis_min_pv_gene: Optional[pd.Series] = None
        self.trans_min_pv_snps: Optional[pd.Series] = None
        self.trans_min_pv_gene: Optional[pd.Series] = None
        # histograms
        self.all_hist_bins: Optional[np.ndarray] = None
        self.all_hist_counts: Optional[np.ndarray] = None
        self.cis_hist_bins: Optional[np.ndarray] = None
        self.cis_hist_counts: Optional[np.ndarray] = None
        self.trans_hist_bins: Optional[np.ndarray] = None
        self.trans_hist_counts: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # result tables
    # ------------------------------------------------------------------
    @property
    def all(self) -> pd.DataFrame:
        """All-pairs eQTL table (populated when there is no cis/trans split)."""
        return self._all if self._all is not None else _EMPTY.copy()

    @property
    def cis(self) -> pd.DataFrame:
        """cis (local) eQTL table."""
        return self._cis if self._cis is not None else _EMPTY.copy()

    @property
    def trans(self) -> pd.DataFrame:
        """trans (distant) eQTL table."""
        return self._trans if self._trans is not None else _EMPTY.copy()

    @property
    def eqtls(self) -> pd.DataFrame:
        """Best-available result table: ``all``, else ``trans`` + ``cis``."""
        if self._all is not None:
            return self._all
        frames = [f for f in (self._cis, self._trans) if f is not None]
        if not frames:
            return _EMPTY.copy()
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True)

    # R-compatible attribute aliases ----------------------------------
    @property
    def time(self):  # noqa: D401 - alias
        """Alias for :attr:`time_in_sec` (R ``$time.in.sec``)."""
        return self.time_in_sec

    def summary(self) -> str:
        """Short human-readable summary of the run."""
        lines = [
            f"MatrixEQTL result ({self.param.get('useModelName', '?')})",
            f"  runtime: {self.time_in_sec:.3f} s",
        ]
        if self._all is not None:
            lines.append(
                f"  all:   {len(self._all)} eQTLs / {self.all_ntests} tests"
            )
        if self._cis is not None:
            lines.append(
                f"  cis:   {len(self._cis)} eQTLs / {self.cis_ntests} tests"
            )
        if self._trans is not None:
            lines.append(
                f"  trans: {len(self._trans)} eQTLs / {self.trans_ntests} tests"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
