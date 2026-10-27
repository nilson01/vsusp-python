# Methods and limitations

## Coefficient-draw selection

Use a finite matrix with draws in rows and coefficients in columns. Remove the
intercept. Keep coefficient units unchanged when comparing thresholds.

S2M follows section 2.2.1 of [Li and Pati (2017)](https://doi.org/10.1016/j.csda.2016.10.008).
It recursively splits absolute coefficients and retains the saved lower-mean
noise candidate at stopping. The center-gap comparison is strict: gap > b.
The signal count is the mode across draws; tied modes favor the smaller count.

Basic 2-M uses the smaller cluster size, which need not be the lower-mean cluster.
Both rules rank variables by median absolute coefficient, with larger column
indices winning equal-median ties. Sparse spike draws can give zero medians for
all variables; a ranking tie can then select a never-included variable. Use
inclusion summaries when inclusion probability is the target.

`small_set="legacy"` retains recursive candidate pairs. `"refine"` splits a
distinct pair. Initial constant magnitudes have no defined split. The explicit
all-zero convention assigns zero signals; S2M requires `zero_draws="empty"`.
These are boundary choices, not a new selection guarantee. Continuous-prior S2M
draws retain at least one signal. No automatic threshold is provided.

## Gaussian fitting

The model is

$$
Y = X\beta + \epsilon, \qquad \epsilon \sim N(0,\sigma^2 I).
$$

Here Y is the response vector, X is the model matrix, $\beta$ denotes the
coefficient vector and sigma² is the residual variance. The mathematical model
includes an intercept; the fitting functions add it internally.

Predictors and response are centered to integrate a flat intercept prior.
They are not standardized. The coefficient priors below exclude the intercept, whose prior is flat.
The residual variance prior is inverse-gamma with
shape 3/2 and scale 3/2. This intercept-prior convention is explicit; the final
paper does not supply every fitting detail.

| Prior | Coefficient distribution given scales and sigma² |
|---|---|
| Ridge | $N(0,\sigma^2\mathtt{prior\_scale}^2)$ |
| Horseshoe | $N(0,\sigma^2\tau^2\lambda_j^2)$; $\tau,\lambda_j$ independent half-Cauchy(0,1) |
| Spike-and-slab | Point mass at zero or $N(0,\sigma^2\omega_j^2)$; $\omega_j^2\sim\mathrm{IG}(3/2,3/2)$; inclusion probability $\sim\mathrm{Beta}(1,15)$ |

The fit stores separate chain and retained-draw axes. `draws` means retained
draws per chain; `burnin` is additional and is discarded. These differ from the
paper's convention of reporting total iterations. Seeds define local NumPy
generators and do not modify NumPy's global random state.

A coefficient interval describes the fitted posterior under these assumptions.
It is not a repeated-dataset coverage guarantee. These Gaussian fitters do not
handle right-censored outcomes as ordinary complete outcomes. Binary, count
and survival likelihoods are not implemented.

## SAVS and DSS

SAVS retains coefficient j only when $\|X_j\|^2|\hat\beta_j|^3>1$, where
$X_j$ is predictor column j and $\hat\beta_j$ is its supplied estimate.
Equality sets the coefficient to zero. Its threshold depends on predictor units.
See [Ray and Bhattacharya (2018)](https://arxiv.org/abs/1810.09004).

For DSS, let $\bar\beta$ be the supplied estimate and $\lambda$ the `penalty`.
The sparse coefficient vector minimizes

$$
n^{-1}\|X\bar\beta-X\gamma\|_2^2
+\lambda\sum_j\frac{|\gamma_j|}{|\bar\beta_j|}.
$$

For positive $\lambda$, $\bar\beta_j=0$ constrains $\gamma_j=0$.
At $\lambda=0$, the supplied estimate is returned.
This is the [Hahn–Carvalho](https://doi.org/10.1080/01621459.2014.993077)
adaptive-L1 approximation, not exact minimization of their variable-count penalty.
Cyclic coordinate descent reports a relative convex optimality error and warns
when the tolerance is unmet. With dependent predictors, sparse coefficients and
selected sets may not be unique.

## Evidence boundaries

The Python package uses the repaired local R implementation as a reference for
fixed draw summaries. This is distinct from comparison with the released CRAN
version. General selection accuracy, high-dimensional sampler mixing and full
paper reproduction require separate evidence. No unrestricted theorem or
false-discovery guarantee is advertised.

## Numerical limits

Fitting stops if a covariance factorization fails or its estimated reciprocal
condition number is at most 1e-12. It does not clip prior scales or silently
retry a failed chain. Extreme predictor units and prior scales can trigger this
limit. Summary and prediction methods reject nonfinite arithmetic; very wide
ranges may be unrepresentable in float64. Integer arrays are accepted only when
conversion preserves their values.
