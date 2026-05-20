"""Plotting helpers -- port of MatrixEQTL's ``plot.MatrixEQTL``.

MatrixEQTL records an optional p-value histogram / Q-Q profile during the
run (via ``pvalue_hist``).  :func:`plot_matrix_eqtl` renders either a
p-value histogram or a Q-Q plot from a :class:`~pymatrixeqtl.MatrixEQTLResult`,
matching the behaviour of the R ``plot`` method.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .results import MatrixEQTLResult

__all__ = ["plot_matrix_eqtl"]


def _qq_xy(eqtls_pv, hist_bins, hist_counts, ntests):
    """Build observed/expected -log10 p coordinates for a Q-Q curve."""
    obs = []
    exp = []
    # contribution from explicitly stored eQTLs
    if eqtls_pv is not None and len(eqtls_pv) > 0:
        pv = np.sort(np.asarray(eqtls_pv))
        ranks = np.arange(1, len(pv) + 1)
        obs.extend(list(-np.log10(pv)))
        exp.extend(list(-np.log10(ranks / ntests)))
    return np.asarray(exp), np.asarray(obs)


def plot_matrix_eqtl(result: MatrixEQTLResult, ax=None, kind: Optional[str] = None):
    """Plot the recorded p-value distribution of a MatrixEQTL run.

    Parameters
    ----------
    result :
        A :class:`~pymatrixeqtl.MatrixEQTLResult`.  Must have been produced
        with a non-``False`` ``pvalue_hist`` argument.
    ax :
        Optional Matplotlib axes; created if omitted.
    kind :
        ``"hist"`` or ``"qqplot"``.  Defaults to whatever ``pvalue_hist``
        was set to.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    pvhist = result.param.get("pvalue.hist", False)
    if pvhist is False:
        raise ValueError(
            "Cannot plot p-value distribution: re-run with pvalue_hist set."
        )
    if kind is None:
        kind = "qqplot" if pvhist == "qqplot" else "hist"

    if kind == "qqplot":
        ntests = (
            result.all_ntests
            or result.trans_ntests
            or result.cis_ntests
            or 1
        )
        cmax = 0.0
        for tbl, color, label in (
            (result.cis, "tab:red", "cis"),
            (result.trans, "tab:blue", "trans"),
            (result.all, "tab:blue", "all"),
        ):
            if tbl is None or len(tbl) == 0:
                continue
            exp, obs = _qq_xy(tbl["pvalue"].to_numpy(), None, None, ntests)
            if len(obs):
                ax.scatter(exp, obs, s=6, color=color, label=label)
                cmax = max(cmax, float(np.max(obs)), float(np.max(exp)))
        lim = max(cmax, 1.0)
        ax.plot([0, lim], [0, lim], color="gray", lw=1)
        ax.set_xlabel(r"$-\log_{10}(p)$ expected")
        ax.set_ylabel(r"$-\log_{10}(p)$ observed")
        ax.set_title("MatrixEQTL Q-Q plot")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    else:
        bins = result.all_hist_bins
        counts = result.all_hist_counts
        if bins is None:
            bins = result.cis_hist_bins
            counts = result.cis_hist_counts
        if bins is None:
            raise ValueError("No histogram recorded in this result.")
        centers = (bins[:-1] + bins[1:]) / 2.0
        width = np.diff(bins)
        density = counts / counts.sum() / width if counts.sum() else counts
        ax.bar(centers, density, width=width, color="tab:blue",
               edgecolor="white", align="center")
        ax.axhline(1.0, color="gray", lw=1, ls="--")
        ax.set_xlabel("p-value")
        ax.set_ylabel("density")
        ax.set_title("MatrixEQTL p-value histogram")
    return ax
