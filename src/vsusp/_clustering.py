"""One-dimensional two-means backends for absolute coefficient draws."""

import numpy as np


def _scaled_sorted(values):
    order = np.argsort(values, kind="stable")
    z = values[order]
    if len(z) < 2 or z[0] == z[-1]:
        raise ValueError("Clustering requires at least two distinct magnitudes.")
    scale = z[-1]
    scaled = z / scale
    distinct = np.diff(z) > 0
    gaps = np.diff(scaled)
    if np.any(distinct & (gaps == 0)):
        raise ValueError("Distinct magnitudes collapsed during numeric rescaling.")
    if np.any((gaps > 0) & (gaps * gaps == 0)):
        raise ValueError(
            "Magnitude separation is too small for squared-loss clustering."
        )
    return order, z, scaled, scale


def exact_two_means(values):
    """Return low/high labels and centers at the minimum sorted-cut loss.

    Values are finite, nonnegative magnitudes. Stable Welford losses are
    computed after scaling; equal losses favor the earliest distinct cut.
    Numerically collapsed separations raise ValueError instead of silently
    changing the partition.
    """
    values = np.asarray(values, dtype=float)
    order, z, scaled, scale = _scaled_sorted(values)
    n = len(z)
    prefix = np.zeros(n)
    suffix = np.zeros(n)
    mean = scaled[0]
    for i in range(1, n):
        delta = scaled[i] - mean
        prefix[i] = prefix[i - 1] + delta * delta * i / (i + 1)
        mean += delta / (i + 1)
    mean = scaled[-1]
    for i in range(n - 2, -1, -1):
        size = n - i
        delta = scaled[i] - mean
        suffix[i] = suffix[i + 1] + delta * delta * (size - 1) / size
        mean += delta / size
    candidates = np.flatnonzero(np.diff(z) > 0)
    losses = prefix[candidates] + suffix[candidates + 1]
    cut = int(candidates[np.argmin(losses)]) + 1
    labels = np.empty(n, dtype=int)
    labels[order[:cut]] = 0
    labels[order[cut:]] = 1
    centers = np.array([scaled[:cut].mean(), scaled[cut:].mean()]) * scale
    return labels, centers


def lloyd_two_means(values, generator):
    """Run NumPy Lloyd iteration with two distinct random initial centers.

    This is a local optimum, with NumPy RNG and Lloyd updates; it does not
    reproduce R's RNG or its default Hartigan-Wong k-means implementation.
    """
    values = np.asarray(values, dtype=float)
    order, _, scaled, scale = _scaled_sorted(values)
    unique = np.unique(scaled)
    if len(values) == 2:
        labels = np.empty(2, dtype=int)
        labels[order] = [0, 1]
        return labels, scaled * scale
    centers = np.sort(generator.choice(unique, size=2, replace=False))
    previous = None
    for _ in range(300):
        # Arithmetic stays on [0, 1], avoiding overflow in distance squares.
        labels = (np.abs(scaled - centers[1]) < np.abs(scaled - centers[0])).astype(int)
        if not np.any(labels == 0) or not np.any(labels == 1):
            raise ValueError(
                "Lloyd clustering produced an empty cluster at this numeric range."
            )
        new_centers = np.array([scaled[labels == k].mean() for k in (0, 1)])
        if new_centers[0] >= new_centers[1]:
            raise ValueError(
                "Distinct cluster centers collapsed during Lloyd iteration."
            )
        if previous is not None and np.array_equal(labels, previous):
            restored = np.empty(len(values), dtype=int)
            restored[order] = labels
            return restored, new_centers * scale
        previous = labels
        centers = new_centers
    raise RuntimeError("Lloyd two-means did not converge within 300 iterations.")


def split_two_means(values, clustering, generator):
    if clustering == "exact":
        return exact_two_means(values)
    return lloyd_two_means(values, generator)
