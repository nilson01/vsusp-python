"""Native Gaussian shrinkage fits with an explicitly reconstructed flat intercept.

Horseshoe and spike-and-slab priors follow Li and Pati (2017), final version,
DOI 10.1016/j.csda.2016.10.008. The original sampler, intercept prior, scan,
initialization and seeds are unavailable: these are independent reconstructions.
IG(a,b) denotes density proportional to x**(-a-1) exp(-b/x). X and y are
centered internally, never standardized. Integrating the flat intercept leaves
n-1 residual degrees of freedom. Returned intercepts are on the input scale.
No convergence or statistical-validation claim follows from obtaining a fit.
"""

import sys
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import dpocon
from scipy.special import expit
from ._validation import matrix, vector, names, integer, scalar
from .results import GaussianFit


__all__ = ["fit_horseshoe", "fit_spike_slab", "fit_ridge"]


def _finite(value, label, *, positive=False):
    value = np.asarray(value)
    if not np.isfinite(value).all() or (positive and np.any(value <= 0)):
        raise FloatingPointError(
            f"{label} exceeded {'positive ' if positive else ''}finite numeric range."
        )
    return value


def _control(value, label, minimum):
    out = integer(value, label, lower=minimum)
    if out > sys.maxsize:
        raise ValueError(f"{label} exceeds the platform integer limit.")
    return out


def _prepare(X, y, feature_names, draws, burnin, chains):
    X, y = matrix(X), vector(y, "y")
    n, p = X.shape
    if n < 2 or y.size != n:
        raise ValueError("X needs at least two rows and y needs one entry per row.")
    feature_names = names(feature_names, p)
    draws = _control(draws, "draws", 1)
    burnin = _control(burnin, "burnin", 0)
    chains = _control(chains, "chains", 1)
    if draws + burnin > sys.maxsize:
        raise ValueError("draws + burnin exceeds the platform integer limit.")
    xbar, ybar = X.mean(axis=0), y.mean()
    Xc, yc = X - xbar, y - ybar
    s = np.sum(Xc * Xc, axis=0)
    _finite(Xc, "Centered predictors")
    _finite(yc, "Centered response")
    _finite(s, "Predictor sums of squares")
    _finite(yc @ yc, "Response sum of squares")
    if np.any(s == 0):
        raise ValueError(
            "Constant predictor columns are not supported; omit the intercept column."
        )
    return Xc, yc, xbar, ybar, s, feature_names, draws, burnin, chains


def _chain_rngs(random_state, chains):
    # A supplied Generator is deliberately advanced to obtain fresh root entropy.
    # No legacy global numpy.random state is read or changed.
    if isinstance(random_state, np.random.Generator):
        entropy = [
            int(x) for x in random_state.integers(0, 2**32, size=4, dtype=np.uint32)
        ]
        origin = "entropy drawn from supplied Generator"
    elif random_state is None:
        entropy = None
        origin = "operating-system entropy"
    else:
        if isinstance(random_state, (int, np.integer)) and not isinstance(
            random_state, (bool, np.bool_)
        ):
            entropy = int(random_state)
            if entropy < 0:
                raise ValueError(
                    "random_state must be a nonnegative integer or Generator."
                )
        else:
            entropy = integer(random_state, "random_state", lower=0)
        origin = "supplied integer"
    root = np.random.SeedSequence(entropy)
    children = root.spawn(chains)
    seeds = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]
    return [np.random.default_rng(seed) for seed in seeds], {
        "seed_origin": origin,
        "root_entropy": root.entropy,
        "chain_seeds": seeds,
        "chain_spawn_keys": [list(child.spawn_key) for child in children],
        "bit_generator": "PCG64",
        "chain_ids": list(range(1, chains + 1)),
    }


def _inverse_gamma(generator, shape, rate, label):
    _finite(rate, f"{label} rate", positive=True)
    gamma = generator.gamma(shape)
    _finite(gamma, f"{label} gamma draw", positive=True)
    value = rate / gamma
    _finite(value, label, positive=True)
    return value


def _gaussian_factor(Xc, yc, d):
    """Factor I+Xc diag(d) Xc'; return factor and marginal IG scale."""
    _finite(d, "Prior variance multipliers", positive=True)
    XD = Xc * np.sqrt(d)
    C = XD @ XD.T + np.eye(len(yc))
    _finite(C, "Gaussian covariance")
    factor = cho_factor(C, lower=True, check_finite=True)
    anorm = float(np.linalg.norm(C, 1))
    _finite(anorm, "Gaussian covariance norm", positive=True)
    reciprocal_condition, info = dpocon(factor[0], anorm, uplo="L")
    if (
        info != 0
        or not np.isfinite(reciprocal_condition)
        or reciprocal_condition <= 1e-12
    ):
        raise FloatingPointError(
            "Gaussian covariance is numerically ill-conditioned; inspect predictor units and prior scales."
        )
    quadratic = float(yc @ cho_solve(factor, yc, check_finite=True))
    _finite(quadratic, "Marginal quadratic")
    if quadratic < 0:
        raise FloatingPointError(
            "Marginal quadratic became negative; numerical factorization is unreliable."
        )
    return factor, 1.5 + quadratic / 2


def _gaussian_block(Xc, yc, d, generator, factor=None, rate=None):
    """Joint sigma2/beta draw: marginal variance first, then new-variance beta.

    The coefficient draw uses the low-rank Gaussian identity of Bhattacharya,
    Chakraborty and Mallick (2016), Algorithm 1. The variance prior here is
    IG(1.5,1.5), so its marginal shape is 1.5+(n-1)/2.
    """
    if factor is None:
        factor, rate = _gaussian_factor(Xc, yc, d)
    sigma2 = float(_inverse_gamma(generator, 1.5 + (len(yc) - 1) / 2, rate, "sigma2"))
    u = np.sqrt(d) * generator.normal(size=Xc.shape[1])
    delta = generator.normal(size=len(yc))
    w = cho_solve(factor, yc / np.sqrt(sigma2) - Xc @ u - delta, check_finite=True)
    beta = np.sqrt(sigma2) * (u + d * (Xc.T @ w))
    _finite(beta, "Gaussian coefficients")
    return beta, sigma2


def _intercept(generator, xbar, ybar, beta, sigma2, n):
    variance = sigma2 / n
    _finite(variance, "Conditional intercept variance", positive=True)
    value = generator.normal(ybar - xbar @ beta, np.sqrt(variance))
    _finite(value, "Conditional intercept")
    return value


def _arrays(chains, draws, p):
    return (
        np.empty((chains, draws, p)),
        np.empty((chains, draws)),
        np.empty((chains, draws)),
    )


def _metadata(
    Xc, xbar, ybar, draws, burnin, chains, seed_info, algorithm, initialization
):
    return {
        **seed_info,
        "n": Xc.shape[0],
        "p": Xc.shape[1],
        "chains": chains,
        "draws_retained_per_chain": draws,
        "burnin_per_chain": burnin,
        "total_iterations_per_chain": draws + burnin,
        "thin": 1,
        "retained_iteration_indices": list(range(burnin + 1, burnin + draws + 1)),
        "draws_meaning": "retained draws per chain; burnin is additional",
        "intercept_prior": "flat; integrated during coefficient/variance updates",
        "centered": True,
        "standardized": False,
        "x_mean": xbar.copy(),
        "y_mean": float(ybar),
        "sigma2_prior": "IG(shape=1.5,scale=1.5)",
        "algorithm": algorithm,
        "initialization": initialization,
        "source": "Li and Pati (2017), final publisher version",
        "doi": "10.1016/j.csda.2016.10.008",
        "reconstruction": "Flat intercept, scan, initialization and seeds are independent choices; original sampler unavailable.",
        "convergence": "not assessed; retained draws do not establish convergence",
        "numpy_version": np.__version__,
    }


def fit_horseshoe(
    X, y, *, draws=1000, burnin=1000, chains=4, random_state=None, feature_names=None
):
    """Fit Gaussian horseshoe regression with genuine, separately seeded chains.

    beta_j | sigma2,tau2,lambda2_j ~ N(0,sigma2*tau2*lambda2_j);
    tau and lambda_j have independent half-Cauchy(0,1) priors, and sigma2
    has IG(1.5,1.5) prior. ``draws`` is retained iterations per chain.
    Initial local/global scales and auxiliary variables equal one. No thinning
    or adaptation is applied. Squared scales are returned in ``extra``.
    Inputs must be finite, with at least two rows and no constant columns.
    Numerical overflow or nonpositive scales fail explicitly without clipping.
    """
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        Xc, yc, xbar, ybar, s, feature_names, draws, burnin, chains = _prepare(
            X, y, feature_names, draws, burnin, chains
        )
        n, p = Xc.shape
        generators, seed_info = _chain_rngs(random_state, chains)
        betas, intercepts, sigmas = _arrays(chains, draws, p)
        lambdas, taus = np.empty_like(betas), np.empty_like(sigmas)
        for c, generator in enumerate(generators):
            lambda2, nu = np.ones(p), np.ones(p)
            tau2 = xi = 1.0
            for iteration in range(draws + burnin):
                beta, sigma2 = _gaussian_block(Xc, yc, tau2 * lambda2, generator)
                lambda2 = _inverse_gamma(
                    generator,
                    np.ones(p),
                    1 / nu + (beta / np.sqrt(sigma2 * tau2)) ** 2 / 2,
                    "lambda2",
                )
                nu = _inverse_gamma(generator, np.ones(p), 1 + 1 / lambda2, "nu")
                tau2 = float(
                    _inverse_gamma(
                        generator,
                        (p + 1) / 2,
                        1 / xi + np.sum((beta / np.sqrt(lambda2)) ** 2) / (2 * sigma2),
                        "tau2",
                    )
                )
                xi = float(_inverse_gamma(generator, 1.0, 1 + 1 / tau2, "xi"))
                if iteration >= burnin:
                    k = iteration - burnin
                    betas[c, k], sigmas[c, k] = beta, sigma2
                    lambdas[c, k], taus[c, k] = lambda2, tau2
                    intercepts[c, k] = _intercept(
                        generator, xbar, ybar, beta, sigma2, n
                    )
        metadata = _metadata(
            Xc,
            xbar,
            ybar,
            draws,
            burnin,
            chains,
            seed_info,
            "joint marginal sigma2/beta block; auxiliary inverse-gamma horseshoe Gibbs updates",
            {"lambda2": 1.0, "nu": 1.0, "tau2": 1.0, "xi": 1.0},
        )
        return GaussianFit(
            betas,
            intercepts,
            sigmas,
            "horseshoe",
            feature_names,
            {"lambda2": lambdas, "tau2": taus},
            metadata,
        )


def fit_spike_slab(
    X, y, *, draws=1000, burnin=1000, chains=4, random_state=None, feature_names=None
):
    """Fit Gaussian exact-zero spike-and-slab regression.

    Inclusion probability q has Beta(1,15) prior. Included beta_j has
    N(0,sigma2*omega2_j) prior, with independent omega2_j ~ IG(1.5,1.5);
    excluded beta_j is exactly zero. sigma2 ~ IG(1.5,1.5). Each fixed-order
    coordinate sweep jointly updates inclusion and its coefficient after
    integrating that coefficient for the inclusion probability. The variance
    full conditional has shape 1.5+(n-1+k)/2 for k active coefficients.
    ``draws`` is retained iterations per chain. No thinning or adaptation.
    ``extra`` retains included, q and omega2 arrays. No constant columns.
    Numerical overflow, rounded-zero slab draws or boundary q draws fail.
    """
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        Xc, yc, xbar, ybar, s, feature_names, draws, burnin, chains = _prepare(
            X, y, feature_names, draws, burnin, chains
        )
        n, p = Xc.shape
        generators, seed_info = _chain_rngs(random_state, chains)
        betas, intercepts, sigmas = _arrays(chains, draws, p)
        included, omega_draws = np.empty(betas.shape, dtype=bool), np.empty_like(betas)
        q_draws = np.empty_like(sigmas)
        for c, generator in enumerate(generators):
            beta, z, omega2 = np.zeros(p), np.zeros(p, dtype=bool), np.ones(p)
            sigma2, q = 1.0, 1 / 16
            for iteration in range(draws + burnin):
                residual = yc - Xc @ beta
                for j in range(p):
                    r = residual + Xc[:, j] * beta[j]
                    t = Xc[:, j] @ r
                    d = s[j] + 1 / omega2[j]
                    variance = sigma2 / d
                    _finite(d, "Coordinate precision", positive=True)
                    _finite(variance, "Coordinate variance", positive=True)
                    score = (t / np.sqrt(d)) / np.sqrt(sigma2)
                    log_bf = -0.5 * np.log1p(omega2[j] * s[j]) + 0.5 * score**2
                    log_odds = np.log(q) - np.log1p(-q) + log_bf
                    _finite(log_odds, "Coordinate log odds")
                    z[j] = generator.random() < expit(log_odds)
                    if z[j]:
                        beta[j] = generator.normal(t / d, np.sqrt(variance))
                        _finite(beta[j], "Included coefficient")
                        if beta[j] == 0:
                            raise FloatingPointError(
                                "Included slab draw rounded to exact zero."
                            )
                    else:
                        beta[j] = 0.0
                    residual = r - Xc[:, j] * beta[j]
                omega2 = _inverse_gamma(
                    generator,
                    1.5 + 0.5 * z,
                    1.5 + 0.5 * (beta / np.sqrt(sigma2)) ** 2,
                    "omega2",
                )
                active = int(z.sum())
                q = generator.beta(1 + active, 15 + p - active)
                if not np.isfinite(q) or not 0 < q < 1:
                    raise FloatingPointError("q draw reached a numerical boundary.")
                residual = yc - Xc @ beta
                rate = 1.5 + 0.5 * (
                    residual @ residual + np.sum((beta[z] / np.sqrt(omega2[z])) ** 2)
                )
                sigma2 = float(
                    _inverse_gamma(
                        generator, 1.5 + (n - 1 + active) / 2, rate, "sigma2"
                    )
                )
                intercept = _intercept(generator, xbar, ybar, beta, sigma2, n)
                if iteration >= burnin:
                    k = iteration - burnin
                    betas[c, k], included[c, k], omega_draws[c, k] = beta, z, omega2
                    sigmas[c, k], q_draws[c, k], intercepts[c, k] = sigma2, q, intercept
        metadata = _metadata(
            Xc,
            xbar,
            ybar,
            draws,
            burnin,
            chains,
            seed_info,
            "fixed coordinate order; paired collapsed inclusion/coefficient Gibbs updates",
            {"beta": 0.0, "included": False, "omega2": 1.0, "sigma2": 1.0, "q": 1 / 16},
        )
        metadata["q_prior"] = "Beta(1,15)"
        metadata["omega2_prior"] = "IG(shape=1.5,scale=1.5)"
        return GaussianFit(
            betas,
            intercepts,
            sigmas,
            "spike_slab",
            feature_names,
            {"included": included, "q": q_draws, "omega2": omega_draws},
            metadata,
        )


def fit_ridge(
    X,
    y,
    *,
    draws=1000,
    chains=4,
    prior_scale=1.0,
    random_state=None,
    feature_names=None,
):
    """Independent conjugate Gaussian posterior draws; no burn-in is needed.

    beta_j | sigma2 ~ N(0,sigma2*prior_scale**2), sigma2 ~ IG(1.5,1.5),
    with the same reconstructed flat intercept as the other Gaussian fits.
    ``prior_scale`` must be strictly positive and finite. ``draws`` means
    retained draws per independently seeded stream. Ridge is a conjugate
    comparison model, not a prior claimed for Li and Pati's horseshoe/SS study.
    Inputs need at least two rows and no constant predictor columns.
    """
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        prior_scale = scalar(prior_scale, "prior_scale")
        if prior_scale <= 0:
            raise ValueError("prior_scale must be strictly positive.")
        Xc, yc, xbar, ybar, s, feature_names, draws, burnin, chains = _prepare(
            X, y, feature_names, draws, 0, chains
        )
        n, p = Xc.shape
        d = np.full(p, prior_scale) ** 2
        factor, rate = _gaussian_factor(Xc, yc, d)
        generators, seed_info = _chain_rngs(random_state, chains)
        betas, intercepts, sigmas = _arrays(chains, draws, p)
        for c, generator in enumerate(generators):
            for k in range(draws):
                beta, sigma2 = _gaussian_block(Xc, yc, d, generator, factor, rate)
                betas[c, k], sigmas[c, k] = beta, sigma2
                intercepts[c, k] = _intercept(generator, xbar, ybar, beta, sigma2, n)
        metadata = _metadata(
            Xc,
            xbar,
            ybar,
            draws,
            0,
            chains,
            seed_info,
            "independent conjugate marginal sigma2/beta draws",
            "none; iid draws",
        )
        metadata["prior_scale"] = prior_scale
        metadata["source"] = (
            "Conjugate Gaussian comparison model with IG(1.5,1.5) variance prior"
        )
        metadata["doi"] = None
        return GaussianFit(
            betas, intercepts, sigmas, "ridge", feature_names, {}, metadata
        )
