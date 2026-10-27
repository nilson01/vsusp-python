"""Shared numeric and shape contracts."""

import numpy as np


def _float64(raw, name):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = np.asarray(raw, dtype=float)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite float64 values.")
    if raw.dtype.kind in "iu":
        large = raw > 2**53
        if raw.dtype.kind == "i":
            large = large | (raw < -(2**53))
        if any(int(v) != float(v) for v in raw[large].flat):
            raise ValueError(f"{name} loses integer precision in float64 conversion.")
    elif np.any(raw != out.astype(raw.dtype)):
        raise ValueError(f"{name} loses precision in float64 conversion.")
    return out


def matrix(value, name="X", min_cols=1):
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.shape[0] < 1
        or raw.shape[1] < min_cols
        or raw.dtype.kind not in "iuf"
    ):
        raise ValueError(
            f"{name} must be a finite real matrix with at least {min_cols} columns."
        )
    return _float64(raw, name)


def vector(value, name="beta"):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty finite real vector.")
    return _float64(raw, name)


def scalar(value, name, lower=None, upper=None):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf" or not np.isfinite(raw):
        raise ValueError(f"{name} must be a finite real number.")
    out = float(_float64(raw, name))
    if (lower is not None and out < lower) or (upper is not None and out > upper):
        raise ValueError(f"{name} is outside the allowed range.")
    return out


def integer(value, name, lower=0):
    raw = np.asarray(value)
    if raw.ndim == 0 and raw.dtype.kind in "iu":
        out = int(raw)
        if out < lower:
            raise ValueError(f"{name} is outside the allowed range.")
        return out
    out = scalar(value, name, lower)
    if out != int(out):
        raise ValueError(f"{name} must be an integer.")
    return int(out)


def names(feature_names, p):
    if feature_names is None:
        return tuple(f"x{j + 1}" for j in range(p))
    if isinstance(feature_names, str):
        raise ValueError("feature_names must contain one name per column.")
    out = tuple(feature_names)
    if (
        len(out) != p
        or any(not isinstance(x, str) or not x for x in out)
        or len(set(out)) != p
    ):
        raise ValueError(
            "feature_names must be unique nonempty strings, one per column."
        )
    return out


def rng(random_state=None):
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is not None:
        random_state = integer(random_state, "random_state")
    return np.random.default_rng(random_state)
