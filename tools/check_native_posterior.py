"""Fixed posterior checks against independent numerical integration.

No private data. Quadrature uses coefficient-space completion of squares;
the production sampler uses observation-space conditional draws.
"""

import argparse
import hashlib
import itertools
import json
import platform

try:
    import resource
except ImportError:
    resource = None
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad_vec
from scipy.special import betaln, gammaln, logsumexp
from numpy.polynomial.legendre import leggauss
from vsusp import fit_horseshoe, fit_spike_slab, diagnostics


def conditional(X, y, variance):
    """Integrated sigma2, flat intercept; moments conditional on prior scales."""
    p = X.shape[1]
    a = 1.5 + (len(y) - 1) / 2
    b0 = 1.5 + y @ y / 2
    if p == 0:
        return 0.0, np.empty(0), np.empty(0), b0 / (a - 1)
    root = np.sqrt(variance)
    # Parameterize beta=diag(root)*standardized coefficient to avoid huge precision.
    C = np.eye(p) + (root[:, None] * (X.T @ X)) * root[None, :]
    rhs = root * (X.T @ y)
    mean = root * np.linalg.solve(C, rhs)
    cov = (root[:, None] * np.linalg.inv(C)) * root[None, :]
    b = b0 - (X.T @ y) @ mean / 2
    ev = b / (a - 1)
    logbf = -np.linalg.slogdet(C)[1] / 2 + a * (np.log(b0) - np.log(b))
    return logbf, mean, mean**2 + np.diag(cov) * ev, ev


def hs_reference(X, y, bound=24):
    """p=1: convolution of log-half-Cauchy priors reduces scales to one integral."""
    X = X - X.mean(0)
    y = y - y.mean()
    assert X.shape[1] == 1

    def f(u):
        logbf, mean, second, ev = conditional(X, y, np.array([np.exp(2 * u)]))
        prior = 2 / np.pi**2 if abs(u) < 1e-7 else 2 * u / (np.pi**2 * np.sinh(u))
        return prior * np.exp(logbf) * np.array([1.0, mean[0], second[0], ev])

    val, err = quad_vec(f, -bound, bound, epsabs=1e-10, epsrel=1e-10)
    return val[1:] / val[0], float(err / val[0])


def ss_reference(X, y, order=100, lower=-8.0, upper=22.0):
    """p=2 model enumeration; independent IG slab scales with log-Jacobians."""
    X = X - X.mean(0)
    y = y - y.mean()
    p = X.shape[1]
    nodes, weights = leggauss(order)
    u = (upper - lower) * nodes / 2 + (upper + lower) / 2
    w = weights * (upper - lower) / 2
    logprior = 1.5 * np.log(1.5) - gammaln(1.5) - 1.5 * u - 1.5 * np.exp(-u)
    records = []
    for mask in itertools.product([0, 1], repeat=p):
        indices = np.flatnonzero(mask)
        k = len(indices)
        logmp = betaln(1 + k, 15 + p - k) - betaln(1, 15)
        if k == 0:
            _, _, _, ev = conditional(X[:, indices], y, np.empty(0))
            records.append((mask, logmp, np.zeros(p), np.zeros(p), ev))
            continue
        coords = np.array(list(itertools.product(range(order), repeat=k)))
        logweights = []
        means = []
        seconds = []
        variances = []
        for ix in coords:
            logbf, m, s, v = conditional(X[:, indices], y, np.exp(u[ix]))
            mm = np.zeros(p)
            ss = np.zeros(p)
            mm[indices] = m
            ss[indices] = s
            logweights.append(logbf + np.sum(logprior[ix] + np.log(w[ix])))
            means.append(mm)
            seconds.append(ss)
            variances.append(v)
        logweights = np.array(logweights)
        normalizer = logsumexp(logweights)
        prob = np.exp(logweights - normalizer)
        records.append(
            (
                mask,
                logmp + normalizer,
                prob @ np.array(means),
                prob @ np.array(seconds),
                float(prob @ variances),
            )
        )
    probs = np.exp(
        np.array([r[1] for r in records]) - logsumexp([r[1] for r in records])
    )
    targets = np.r_[
        probs @ np.array([r[2] for r in records]),
        probs @ np.array([r[3] for r in records]),
        probs @ np.array([r[4] for r in records]),
        probs @ np.array([r[0] for r in records]),
        probs,
    ]
    return targets


def assess(data, target):
    diag = diagnostics(data)
    estimate = data.mean((0, 1))
    batch = {}
    for size in [100, 200]:
        b = (
            data.reshape(data.shape[0], data.shape[1] // size, size, data.shape[2])
            .mean(2)
            .reshape(-1, data.shape[2])
        )
        batch[str(size)] = (b.std(0, ddof=1) / np.sqrt(len(b))).tolist()
    se = np.maximum.reduce(
        [diag.mcse_mean, np.array(batch["100"]), np.array(batch["200"])]
    )
    tolerance = np.maximum(0.01, 5 * se)
    diagnostic_pass = (
        np.isfinite(diag.rhat)
        & (diag.rhat <= 1.05)
        & (diag.ess_bulk >= 100)
        & (diag.ess_tail >= 100)
    )
    differences = estimate - target
    return {
        "target": target.tolist(),
        "estimate": estimate.tolist(),
        "difference": differences.tolist(),
        "rhat": diag.rhat.tolist(),
        "ess_bulk": diag.ess_bulk.tolist(),
        "ess_tail": diag.ess_tail.tolist(),
        "diagnostic_flags": list(diag.flags),
        "mean_mcse": diag.mcse_mean.tolist(),
        "batch_mcse": batch,
        "tolerance": tolerance.tolist(),
        "moment_pass": (abs(differences) <= tolerance).tolist(),
        "diagnostic_pass": diagnostic_pass.tolist(),
        "pass": bool(np.all(abs(differences) <= tolerance) & np.all(diagnostic_pass)),
    }


def cases():
    x = np.arange(-3.0, 4.0)
    e = np.array([0.3, -0.4, 0.2, -0.1, 0.3, -0.2, -0.1])
    z = np.array([1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0])
    return [
        ("hs_weak", "hs", x[:, None], 0.12 * x + e, 2901),
        ("hs_strong", "hs", x[:, None], 1.6 * x + e, 2902),
        ("ss_orthogonal", "ss", np.column_stack([x, z]), 0.75 * x + e, 2903),
        ("ss_correlated", "ss", np.column_stack([x, x + 0.5 * z]), 0.75 * x + e, 2904),
    ]


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    start = time.perf_counter()
    if args.timing:
        t = time.perf_counter()
        fit_horseshoe(
            cases()[0][2],
            cases()[0][3],
            draws=100,
            burnin=50,
            chains=1,
            random_state=2900,
        )
        print(json.dumps({"iterations": 150, "seconds": time.perf_counter() - t}))
        return
    for name, kind, X, y, seed in cases():
        try:
            t = time.perf_counter()
            if kind == "hs":
                low, err1 = hs_reference(X, y, 24)
                target, err2 = hs_reference(X, y, 30)
                fit = fit_horseshoe(
                    X, y, draws=4000, burnin=2000, chains=4, random_state=seed
                )
                data = np.concatenate(
                    [
                        fit.coefficient_draws,
                        fit.coefficient_draws**2,
                        fit.sigma2_draws[:, :, None],
                    ],
                    axis=-1,
                )
                labels = ["beta1", "beta1_squared", "sigma2"]
                qdiff = float(np.max(abs(target - low)))
            else:
                low = ss_reference(X, y, 100)
                target = ss_reference(X, y, 160, lower=-9, upper=25)
                qdiff = float(np.max(abs(target - low)))
                fit = fit_spike_slab(
                    X, y, draws=4000, burnin=2000, chains=4, random_state=seed
                )
                z = fit.extra["included"].astype(float)
                model = np.stack(
                    [
                        np.all(z == mask, axis=-1)
                        for mask in itertools.product([0, 1], repeat=2)
                    ],
                    axis=-1,
                ).astype(float)
                data = np.concatenate(
                    [
                        fit.coefficient_draws,
                        fit.coefficient_draws**2,
                        fit.sigma2_draws[:, :, None],
                        z,
                        model,
                    ],
                    axis=-1,
                )
                labels = [
                    "beta1",
                    "beta2",
                    "beta1_squared",
                    "beta2_squared",
                    "sigma2",
                    "pip1",
                    "pip2",
                    "p00",
                    "p01",
                    "p10",
                    "p11",
                ]
            result = {
                "name": name,
                "family": kind,
                "X": X.tolist(),
                "y": y.tolist(),
                "seed": seed,
                "chain_seeds": fit.metadata["chain_seeds"],
                "labels": labels,
                "quadrature_error_estimates": [err1, err2] if kind == "hs" else None,
                "quadrature_resolution_difference": qdiff,
                **assess(data, target),
                "seconds": time.perf_counter() - t,
            }
            result["pass"] = result["pass"] and qdiff <= 1e-5
            np.savez_compressed(
                args.out / (name + ".npz"),
                coefficients=fit.coefficient_draws,
                sigma2=fit.sigma2_draws,
                intercepts=fit.intercept_draws,
                comparison=data,
            )
            (args.out / (name + ".json")).write_text(
                json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n"
            )
            results.append(result)
            print(
                name,
                result["pass"],
                "quadrature",
                qdiff,
                "seconds",
                result["seconds"],
                flush=True,
            )
        except Exception as exc:
            result = {
                "name": name,
                "family": kind,
                "seed": seed,
                "pass": False,
                "X": X.tolist(),
                "y": y.tolist(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            (args.out / (name + ".json")).write_text(
                json.dumps(result, indent=2) + "\n"
            )
            results.append(result)
            print(name, "FAILED", type(exc).__name__, str(exc), flush=True)
    out = {
        "cases": results,
        "pass": all(r["pass"] for r in results),
        "seconds": time.perf_counter() - start,
        "maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if resource is not None
        else None,
        "maxrss_units": "bytes on macOS; KiB on Linux",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "draws": 4000,
        "burnin": 2000,
        "chains": 4,
    }
    (args.out / "results.json").write_text(
        json.dumps(json_safe(out), indent=2, allow_nan=False) + "\n"
    )
    if not out["pass"]:
        raise SystemExit(
            "At least one fixed-posterior check failed; inspect retained results."
        )


if __name__ == "__main__":
    main()
