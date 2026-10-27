"""Sparse summaries of supplied coefficient estimates."""

import warnings

import numpy as np

from ._validation import integer, matrix, names, scalar, vector
from .results import SparseResult


def _inputs(X, beta, feature_names):
    X = matrix(X)
    beta = vector(beta)
    if X.shape[1] != beta.size:
        raise ValueError("beta must have one coefficient per column of X.")
    labels = names(feature_names, beta.size)
    magnitudes = np.max(np.abs(X), axis=0)
    if np.any(magnitudes == 0):
        raise ValueError("X must not contain zero-norm columns.")
    return X, beta, labels, magnitudes


def savs(X, beta, *, feature_names=None):
    """Apply the closed-form signal adaptive variable selector (SAVS).

    X is a finite observation-by-predictor matrix and beta is a finite point
    estimate in the same column order. Exclude the intercept and zero-norm
    columns. No centering or scaling is performed; selection depends on units.
    A coefficient survives only when sum(X[:, j]**2) * abs(beta[j])**3 > 1.
    Equality gives zero. Logarithms protect the decision from overflow, with a
    conservative rounding band when direct factors cannot be represented.

    This applies equation (2) and Algorithm 1 of Ray and Bhattacharya (2018),
    arXiv:1810.09004. It does not solve the full correlated-design optimization,
    estimate inclusion probabilities, or establish false-discovery control.
    The returned selected indices are zero-based. diagnostics['log_score']
    contains negative infinity for exactly zero input coefficients.
    """
    X, beta, labels, magnitudes = _inputs(X, beta, feature_names)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        normalized = X / magnitudes
        log_norm2 = 2 * np.log(magnitudes) + np.log(np.sum(normalized**2, axis=0))
        log_beta = np.log(np.abs(beta))
        log_score = 3 * log_beta + log_norm2
        norm2 = np.sum(X**2, axis=0)
        cube = np.abs(beta) ** 3
        direct = np.isfinite(norm2) & (norm2 > 0) & np.isfinite(cube) & (cube > 0)
        tolerance = (
            8
            * np.finfo(float).eps
            * np.maximum(1, np.abs(log_norm2) + 3 * np.abs(log_beta))
        )
        keep = log_score > tolerance
        indices = np.flatnonzero(direct)
        score = norm2[indices] * cube[indices]
        keep[indices] = score > 1
        finite_score = np.isfinite(score) & (score > 0)
        log_score[indices[finite_score]] = np.log(score[finite_score])
        coefficients = np.zeros_like(beta)
        coefficients[keep] = beta[keep] * (-np.expm1(-log_score[keep]))
    if not np.isfinite(coefficients).all():
        raise ValueError(
            "SAVS arithmetic is outside the representable range; rescale the inputs."
        )
    return SparseResult(
        coefficients=coefficients,
        selected=np.flatnonzero(coefficients != 0),
        method="SAVS",
        feature_names=labels,
        diagnostics={"log_score": log_score},
        metadata={
            "p": beta.size,
            "source": "Ray and Bhattacharya (2018), arXiv:1810.09004, equation (2) and Algorithm 1",
        },
    )


def _finite(value):
    if not np.isfinite(value).all():
        raise ValueError(
            "DSS arithmetic is outside the representable range; rescale the inputs."
        )
    return value


def _squared(value):
    result = _finite(np.square(value))
    if np.any((value != 0) & (result == 0)):
        raise ValueError("DSS squared errors or norms underflow; rescale the inputs.")
    return result


def _violation(gradient, alpha, penalty):
    return _finite(
        np.where(
            alpha != 0,
            np.abs(gradient - penalty * np.sign(alpha)),
            np.maximum(np.abs(gradient) - penalty, 0),
        )
    )


def dss(X, beta, penalty, *, tol=1e-8, max_iter=10000, feature_names=None):
    """Fit the decoupled shrinkage and selection adaptive-L1 approximation.

    Minimize mean((X @ beta - X @ coefficients)**2) plus penalty times
    sum(abs(coefficients[j]) / abs(beta[j])) over nonzero beta entries.
    For positive penalty, zero beta entries constrain coefficients to zero.
    With penalty=0, beta is returned as a least-squares minimizer. X contains
    the prediction points; exclude the intercept and keep it separately.
    No centering, scaling, intercept, or penalty selection is added.

    Cyclic coordinate descent uses X * abs(beta), in supplied column order.
    Convergence requires the relative Karush-Kuhn-Tucker (KKT) optimality error
    to be at most tol; its denominator is the larger of penalty and the
    largest absolute initial gradient. Inspect converged before using the
    selection. Nonunique coefficients and selected sets are possible with
    dependent columns. Detected overflow and underflow raise ValueError.

    This implements Hahn and Carvalho (2015), equation (21),
    doi:10.1080/01621459.2014.993077: a convex adaptive-L1 approximation, not
    exact minimization of the L0 counting penalty. Approximation MSE is not
    held-out predictive error. No inclusion probabilities or false-discovery
    guarantees are provided. Selected indices are zero-based.
    """
    X, beta, labels, _ = _inputs(X, beta, feature_names)
    penalty = scalar(penalty, "penalty", lower=0)
    tol = scalar(tol, "tol")
    if not 0 < tol < 1:
        raise ValueError("tol must be strictly between zero and one.")
    max_iter = integer(max_iter, "max_iter", lower=1)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        return _dss(X, beta, labels, penalty, tol, max_iter)


def _dss(X, beta, labels, penalty, tol, max_iter):
    n = X.shape[0]
    target = _finite(X @ beta)
    _squared(target)
    active = np.flatnonzero(beta != 0)
    magnitudes = np.abs(beta[active])
    Z = _finite(X[:, active] * magnitudes)
    if np.any((X[:, active] != 0) & (Z == 0)):
        raise ValueError("DSS transformed columns underflow; rescale the inputs.")
    norm2 = _finite(np.mean(_squared(Z), axis=0))
    if np.any(norm2 == 0):
        raise ValueError("DSS transformed column norms underflow; rescale the inputs.")
    gradient0 = _finite(2 * (Z.T @ target) / n)
    initial_max = float(np.max(np.abs(gradient0), initial=0))
    scale = max(penalty, initial_max)
    alpha = np.zeros(active.size)
    iterations = 0
    if penalty == 0:
        alpha = np.sign(beta[active])
    elif active.size and penalty < initial_max:
        threshold = penalty / 2
        if threshold == 0:
            raise ValueError("penalty/2 underflows; rescale the inputs.")
        residual = target.copy()
        for iteration in range(1, max_iter + 1):
            for j in range(active.size):
                partial = _finite(np.sum(Z[:, j] * residual) / n + norm2[j] * alpha[j])
                next_alpha = _finite(
                    np.sign(partial) * max(abs(partial) - threshold, 0) / norm2[j]
                )
                residual = _finite(residual - Z[:, j] * (next_alpha - alpha[j]))
                alpha[j] = next_alpha
            # Recompute to avoid carrying residual drift between sweeps.
            residual = _finite(target - Z @ alpha)
            gradient = _finite(2 * (Z.T @ residual) / n)
            violation = _violation(gradient, alpha, penalty)
            iterations = iteration
            if np.max(violation, initial=0) / scale <= tol:
                break
    coefficients = np.zeros_like(beta)
    coefficients[active] = _finite(magnitudes * alpha)
    if np.any((alpha != 0) & (coefficients[active] == 0)):
        raise ValueError("DSS sparse coefficients underflow; rescale the inputs.")
    fitted = _finite(X @ coefficients)
    residual = _finite(target - fitted)
    effective_alpha = _finite(coefficients[active] / magnitudes)
    if np.any((coefficients[active] != 0) & (effective_alpha == 0)):
        raise ValueError("DSS transformed coefficients underflow; rescale the inputs.")
    gradient = _finite(2 * (Z.T @ residual) / n)
    violation = _violation(gradient, effective_alpha, penalty)
    kkt_error = (
        0.0 if scale == 0 else float(_finite(np.max(violation, initial=0) / scale))
    )
    residual_squared = _squared(residual)
    mse = float(_finite(np.mean(residual_squared)))
    if mse == 0 and np.any(residual_squared != 0):
        raise ValueError("DSS mean squared error underflows; rescale the inputs.")
    weighted_l1 = _finite(np.sum(np.abs(effective_alpha)))
    penalty_term = float(_finite(penalty * weighted_l1))
    objective = float(_finite(mse + penalty_term))
    converged = kkt_error <= tol
    if not converged:
        warnings.warn(
            "DSS did not meet the KKT tolerance; increase max_iter or inspect predictor dependence.",
            RuntimeWarning,
            stacklevel=3,
        )
    return SparseResult(
        coefficients=coefficients,
        selected=np.flatnonzero(coefficients != 0),
        method="DSS",
        feature_names=labels,
        converged=converged,
        iterations=iterations,
        objective=objective,
        diagnostics={
            "fitted": fitted,
            "mse": mse,
            "penalty": penalty_term,
            "kkt_error": kkt_error,
            "kkt_scale": scale,
        },
        metadata={
            "p": beta.size,
            "penalty": penalty,
            "tol": tol,
            "max_iter": max_iter,
            "source": "Hahn and Carvalho (2015), doi:10.1080/01621459.2014.993077, equation (21)",
        },
    )
