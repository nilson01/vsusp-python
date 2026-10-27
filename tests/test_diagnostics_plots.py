import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vsusp import (
    diagnostics,
    fit_ridge,
    s2m,
    spike_slab,
    plot_tuning,
    plot_coefficients,
    plot_inclusion,
    plot_trace,
    plot_autocorrelation,
)


def test_diagnostics_chain_contract_and_unavailability():
    d = np.random.default_rng(71).normal(size=(2, 100, 2))
    out = diagnostics(d)
    assert np.isfinite(out.rhat).all() and (out.ess_bulk > 0).all()
    assert out.backend in ("arviz", "arviz_stats")
    one = diagnostics(d[:1])
    assert np.isnan(one.rhat).all() and "one chain" in one.flags[0]
    constant = diagnostics(np.zeros((2, 20, 1)))
    assert np.isnan(constant.rhat).all() and "constant" in constant.flags[0]
    with pytest.raises(ValueError):
        diagnostics(d.reshape(200, 2))
    with pytest.raises(ValueError):
        diagnostics(d[:, :3])


def test_plot_outputs_and_axis_definitions(tmp_path):
    X = np.array([[-2.0, 0], [-1, 1], [0, -1], [1, 0], [2, 1], [0, -1]])
    y = np.array([-3.8, -2.1, 0.1, 2.2, 3.9, -0.2])
    fit = fit_ridge(X, y, draws=30, chains=2, random_state=8)
    tuning = s2m([[0.0, 0.1, 10], [0, 0.2, 11]], [0.1, 0.5])
    inclusion = spike_slab([[0.0, 1], [0, 0], [1, 0]])
    plots = [
        plot_tuning(tuning),
        plot_coefficients(fit, selected=[0]),
        plot_inclusion(inclusion),
        plot_trace(fit),
        plot_autocorrelation(fit),
    ]
    for i, ax in enumerate(plots):
        ax.figure.savefig(tmp_path / f"plot{i}.svg")
        assert (tmp_path / f"plot{i}.svg").stat().st_size > 1000
        plt.close(ax.figure)
    with pytest.raises(IndexError):
        plot_trace(fit, variable=2)
    with pytest.raises(ValueError):
        plot_coefficients(fit, interval=1)
