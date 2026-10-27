"""One fixed posterior versus quadrature; not a repeated-dataset study."""

import json
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
from vsusp import fit_spike_slab


def main():
    x = np.arange(-2, 3, dtype=float)
    y = 0.8 * x + np.array([0.2, -0.3, 0.1, 0.3, -0.3])
    xc = x - x.mean()
    yc = y - y.mean()
    s = xc @ xc
    t = xc @ yc
    shape = 1.5 + (len(x) - 1) / 2
    rate0 = 1.5 + yc @ yc / 2

    def rate(w):
        return rate0 - 0.5 * w * t * t / (1 + w * s)

    def weight(w):
        logprior = 1.5 * np.log(1.5) - gammaln(1.5) - 2.5 * np.log(w) - 1.5 / w
        return np.exp(
            logprior - 0.5 * np.log1p(w * s) + shape * (np.log(rate0) - np.log(rate(w)))
        )

    def integral(f):
        return quad(f, 0, np.inf, epsabs=1e-10, epsrel=1e-9)[0]

    bf = integral(weight)
    pip = bf / (15 + bf)
    target = np.array(
        [
            pip,
            pip * integral(lambda w: weight(w) * w * t / (1 + w * s)) / bf,
            (1 - pip) * rate0 / (shape - 1)
            + pip * integral(lambda w: weight(w) * rate(w) / (shape - 1)) / bf,
        ]
    )
    fit = fit_spike_slab(
        x[:, None], y, draws=3000, burnin=2000, chains=4, random_state=811
    )
    data = np.stack(
        [
            fit.extra["included"][:, :, 0].astype(float),
            fit.coefficient_draws[:, :, 0],
            fit.sigma2_draws,
        ],
        axis=-1,
    )
    batches = data.reshape(4, 30, 100, 3).mean(axis=2).reshape(120, 3)
    estimate = batches.mean(0)
    mcse = batches.std(0, ddof=1) / np.sqrt(len(batches))
    out = {
        "quantities": ["inclusion", "beta_mean", "sigma2_mean"],
        "quadrature": target.tolist(),
        "mcmc": estimate.tolist(),
        "batch_mcse": mcse.tolist(),
        "difference": (estimate - target).tolist(),
        "chain_seeds": fit.metadata["chain_seeds"],
        "interpretation": "one fixed posterior; 100-draw batch MCSE is approximate",
    }
    print(json.dumps(out, indent=2))
    if not np.all(abs(estimate - target) <= np.maximum(0.03, 6 * mcse)):
        raise AssertionError(
            "Fixed-posterior discrepancy exceeds the stated check tolerance."
        )


if __name__ == "__main__":
    main()
