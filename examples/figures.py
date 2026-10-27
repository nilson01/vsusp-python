"""Render fixed illustrative examples, preserving all plotted coefficient draws."""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from vsusp import (
    fit_ridge,
    fit_horseshoe,
    s2m,
    spike_slab,
    plot_tuning,
    plot_coefficients,
    plot_inclusion,
    plot_trace,
    plot_autocorrelation,
)

out = Path(__file__).resolve().parents[1] / "docs" / "figures"
out.mkdir(parents=True, exist_ok=True)
X = np.array([[-2.0, 0], [-1, 1], [0, -1], [1, 0], [2, 1], [0, -1]])
y = np.array([-3.8, -2.1, 0.1, 2.2, 3.9, -0.2])
ridge = fit_ridge(X, y, draws=200, chains=2, random_state=12)
hs = fit_horseshoe(X, y, draws=200, burnin=100, chains=2, random_state=12)
B = np.array(
    [
        [0.01, 0.02, 0.03, 0.3, 0.31, 10, 11],
        [-0.01, -0.02, -0.03, -0.3, -0.31, -10, -11],
    ]
)
path = s2m(B, [0.1, 0.5])
spike = np.array([[1.0, 0, 0], [2, 0, 0], [0, -1, 0], [0, 0, 0]])
plots = {
    "tuning": plot_tuning(path),
    "coefficients": plot_coefficients(ridge),
    "inclusion": plot_inclusion(spike_slab(spike)),
    "trace": plot_trace(hs),
    "autocorrelation": plot_autocorrelation(hs),
}
for name, ax in plots.items():
    ax.spines[["top", "right"]].set_visible(False)
    ax.figure.savefig(out / f"{name}.svg", metadata={"Date": None})
    ax.figure.savefig(out / f"{name}.png", dpi=180)
    plt.close(ax.figure)
np.savetxt(
    out / "X-y.csv",
    np.column_stack([X, y]),
    delimiter=",",
    header="x1,x2,y",
    comments="",
)
np.savetxt(out / "tuning-draws.csv", B, delimiter=",")
np.savetxt(out / "spike-draws.csv", spike, delimiter=",")
with (out / "posterior-draws.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        ["prior", "chain", "retained_draw", "beta1", "beta2", "intercept", "sigma2"]
    )
    for fit in [ridge, hs]:
        for c in range(2):
            for d in range(200):
                writer.writerow(
                    [
                        fit.prior,
                        c + 1,
                        d + 1,
                        *fit.coefficient_draws[c, d],
                        fit.intercept_draws[c, d],
                        fit.sigma2_draws[c, d],
                    ]
                )
