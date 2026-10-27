"""Small Gaussian calls; short chains illustrate use, not convergence."""

import numpy as np
from vsusp import fit_ridge, fit_horseshoe, fit_spike_slab, diagnostics

X = np.array([[-2, 0], [-1, 1], [0, -1], [1, 0], [2, 1], [0, -1.0]])
y = np.array([-3.8, -2.1, 0.1, 2.2, 3.9, -0.2])
for fitter in [fit_ridge, fit_horseshoe, fit_spike_slab]:
    kwargs = {} if fitter is fit_ridge else {"burnin": 100}
    fit = fitter(X, y, draws=200, chains=2, random_state=12, **kwargs)
    print(fit.summary())
    print(fit.predict(X).mean)
    print(diagnostics(fit).summary())
