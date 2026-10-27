"""Matplotlib displays for selection and posterior draws; no global style changes."""

import numpy as np
from ._validation import integer, scalar
from .diagnostics import chain_array
from .results import GaussianFit, SpikeSlabResult, TuningResult, _statistics


def _axes(ax):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('Install plotting with: pip install "vsusp[plots]"') from exc
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
    return ax


def plot_tuning(result, *, ax=None):
    """Plot S2M counts against b, sorted by threshold; choose no optimum."""
    if not isinstance(result, TuningResult):
        raise TypeError("result must be returned by s2m.")
    ax = _axes(ax)
    order = np.argsort(result.thresholds, kind="stable")
    ax.plot(
        result.thresholds[order], result.counts[order], "o-", color="#0072B2", lw=1.3
    )
    ax.set(xlabel=r"$b$", ylabel=r"$H_b$")
    return ax


def plot_coefficients(fit, *, interval=0.95, selected=None, ax=None):
    """Posterior means and pointwise equal-tail intervals, with a zero line."""
    if not isinstance(fit, GaussianFit):
        raise TypeError("fit must be a GaussianFit.")
    interval = scalar(interval, "interval")
    if not 0 < interval < 1:
        raise ValueError("interval must be strictly between zero and one.")
    p = len(fit.feature_names)
    chosen = set()
    if selected is not None:
        chosen = {integer(j, "selected index") for j in selected}
        if any(j >= p for j in chosen):
            raise ValueError("selected index exceeds coefficient count.")
    mean, lo, hi, _ = _statistics(fit.pooled_draws(), interval)
    ax = _axes(ax)
    for j in range(p):
        color = "#D55E00" if j in chosen else "#0072B2"
        ax.hlines(j, lo[j], hi[j], color=color, lw=1.3)
        ax.plot(mean[j], j, "D" if j in chosen else "o", color=color, ms=4)
    ax.axvline(0, color="0.45", lw=0.8, ls="--")
    ax.set(
        yticks=np.arange(p),
        yticklabels=fit.feature_names,
        xlabel=rf"$\beta_j$ ({interval:.0%} posterior intervals)",
    )
    return ax


def plot_inclusion(result, *, ax=None):
    """Plot exact spike inclusion frequencies; dashed line is the MPM cutoff."""
    if not isinstance(result, SpikeSlabResult):
        raise TypeError("result must be returned by spike_slab.")
    ax = _axes(ax)
    pos = np.arange(len(result.feature_names))
    ax.barh(pos, result.inclusion_probability, color="#0072B2", height=0.65)
    ax.axvline(0.5, color="0.35", ls="--", lw=1)
    ax.set(
        yticks=pos,
        yticklabels=result.feature_names,
        xlim=(0, 1),
        xlabel="Posterior inclusion frequency",
    )
    return ax


def _variable(draws, variable):
    a = chain_array(draws)
    j = integer(variable, "variable")
    if j >= a.shape[2]:
        raise IndexError("variable index is out of range.")
    label = draws.feature_names[j] if isinstance(draws, GaussianFit) else f"x{j + 1}"
    return a[:, :, j], label


def plot_trace(draws, *, variable=0, ax=None):
    """Show each chain separately, indexed by retained iteration."""
    a, label = _variable(draws, variable)
    ax = _axes(ax)
    for c in range(len(a)):
        ax.plot(
            np.arange(1, a.shape[1] + 1),
            a[c],
            lw=0.8,
            ls=("-", "--", "-.", ":")[c % 4],
            label=f"Chain {c + 1}",
        )
    ax.set(xlabel="Retained iteration", ylabel=label)
    ax.legend(frameon=False)
    return ax


def plot_autocorrelation(draws, *, variable=0, max_lag=40, ax=None):
    """Show within-chain sample ACF; mark constant chains as unavailable."""
    a, label = _variable(draws, variable)
    if a.shape[1] < 2:
        raise ValueError("At least two retained draws are required.")
    lag = min(integer(max_lag, "max_lag"), a.shape[1] - 1)
    ax = _axes(ax)
    for c, row in enumerate(a):
        scale = np.max(np.abs(row))
        scaled = row / scale if scale else row
        centered = scaled - scaled.mean()
        norm = centered @ centered
        if norm == 0:
            ax.plot([], [], label=f"Chain {c + 1}: constant")
            continue
        acf = (
            np.array([centered[: len(row) - k] @ centered[k:] for k in range(lag + 1)])
            / norm
        )
        ax.plot(
            np.arange(lag + 1),
            acf,
            lw=1,
            ls=("-", "--", "-.", ":")[c % 4],
            label=f"Chain {c + 1}",
        )
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set(xlabel="Lag", ylabel=f"ACF: {label}", ylim=(-1.05, 1.05))
    ax.legend(frameon=False)
    return ax
