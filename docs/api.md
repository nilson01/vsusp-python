# API reference

## Gaussian models

Here n is the number of observations and p is the number of predictors.

| Function | Required inputs | Main controls | Result |
|---|---|---|---|
| `fit_ridge` | X (n × p), y (n) | draws, chains, prior_scale, random_state, feature_names | GaussianFit |
| `fit_horseshoe` | X, y | draws, burnin, chains, random_state, feature_names | GaussianFit |
| `fit_spike_slab` | X, y | draws, burnin, chains, random_state, feature_names | GaussianFit |

Fitting requires at least two observations, finite numeric inputs and no
constant predictor columns. Missing values are rejected. No intercept column
should be added. `GaussianFit.coefficient_draws` has axes (chain, draw, predictor).
`intercept_draws` and `sigma2_draws` have axes (chain, draw). `extra` stores prior
scales and inclusion states; `metadata` records seeds and sampling controls.

`fit.summary(interval=.95)` returns a text table. `fit.pooled_draws()` explicitly
combines chains for selection. `fit.predict(X, interval=.95, include_noise=False)`
returns mean, lower, upper, interval and kind; intervals are pointwise posterior
intervals. The point estimate is the posterior conditional mean; observation
noise affects the intervals, not that point estimate. Preserve predictor order when predicting.

## Posterior selection

| Function | Required inputs | Main controls | Result |
|---|---|---|---|
| `s2m` | draws, thresholds | clustering, small_set, zero_draws, random_state, feature_names | TuningResult |
| `two_means` | draws | clustering, random_state, feature_names | SelectionResult |
| `select` | draws, count | feature_names | SelectionResult |
| `spike_slab` | draws | feature_names | SpikeSlabResult |

A draw matrix has axes (draw, predictor). Exclude the intercept. Clustering
requires at least two predictors; the other summaries allow one. Exact
clustering is the default; `clustering="kmeans"` uses NumPy Lloyd iterations.

`TuningResult` contains thresholds, modal counts, per-draw counts, modal
frequencies, selected sets and adjacent-set Jaccard similarities. `select(i)`
uses the threshold at zero-based position i. `SelectionResult` reports selected
indices, names, median absolute coefficients and per-draw counts where estimated.

`SpikeSlabResult` contains inclusion_probability, mpm, hppm, visited_models,
model_counts and hppm_ties. Zero means an exact spike; no numerical tolerance
converts continuous coefficients to exclusions. Each result has `summary()`.

## Sparse coefficients

| Function | Required inputs | Controls | Result |
|---|---|---|---|
| `savs` | X, beta | feature_names | SparseResult |
| `dss` | X, beta, penalty | tol, max_iter, feature_names | SparseResult |

X must have one nonzero-norm column per beta entry. No centering is added.
The result contains coefficients, selected, converged, iterations, objective,
diagnostics and metadata. SAVS does not optimize a DSS objective; its objective
field is unavailable (NaN). `predict(X, intercept=0)` returns X times the sparse
coefficients plus the supplied intercept. `summary()` returns a text summary.

## Diagnostics and plotting

`diagnostics(fit)` includes beta, intercept and sigma². For external arrays,
use `diagnostics(draws, feature_names=...)` with explicit (chain, draw, predictor)
axes and at least four retained draws. Results contain rank R-hat, bulk/tail ESS,
mean MCSE, flags and backend, plus `summary()`. ESS is effective sample size;
MCSE estimates the numerical error of a sampled posterior mean.

| Plot | Input | Other controls |
|---|---|---|
| `plot_tuning` | TuningResult | ax |
| `plot_coefficients` | GaussianFit | interval, selected, ax |
| `plot_inclusion` | SpikeSlabResult | ax |
| `plot_trace` | fit or chain array | variable, ax |
| `plot_autocorrelation` | fit or chain array | variable, max_lag, ax |

All plots return axes. Variable and selected indices are zero-based. Coefficient
plots show posterior means and equal-tail intervals, not selection-error rates.
