"""Posterior-draw selection from Li and Pati (2017), doi:10.1016/j.csda.2016.10.008."""

import numpy as np

from ._clustering import split_two_means
from ._validation import integer, matrix, names, rng, vector
from .results import SelectionResult, SpikeSlabResult, TuningResult, _rank


def _choice(value, allowed, name):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}.")
    return value


def _medians(magnitudes):
    # Average the central pair without overflowing when both values are large.
    ordered = np.sort(magnitudes, axis=0)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid].copy()
    lower, upper = ordered[mid - 1], ordered[mid]
    return lower + (upper - lower) / 2


def _mode(counts):
    values, frequencies = np.unique(counts, return_counts=True)
    return int(values[np.argmax(frequencies)])


def _frequencies(counts):
    values, frequencies = np.unique(counts, return_counts=True)
    return {
        "counts": values,
        "frequencies": frequencies,
        "proportions": frequencies / len(counts),
    }


def select(draws, count, *, feature_names=None):
    """Select count columns by median absolute coefficient, largest first.

    Rows are posterior draws and columns are coefficients; exclude any
    intercept. Indices are zero-based. Equal medians favor larger indices.
    This supplied-count rule fits no model and estimates no signal count.
    """
    draws = matrix(draws, "draws")
    feature_names = names(feature_names, draws.shape[1])
    count = integer(count, "count")
    if count > draws.shape[1]:
        raise ValueError("count must not exceed the number of coefficient columns.")
    medians = _medians(np.abs(draws))
    return SelectionResult(
        _rank(medians, count),
        "median ranking",
        medians,
        np.empty(0, dtype=int),
        feature_names,
        {"count_source": "supplied", "index_base": 0},
    )


def two_means(draws, *, clustering="exact", random_state=None, feature_names=None):
    """Apply the paper's 2M rule using the smaller cluster size in each draw.

    Rows are draws, columns are coefficients, and any intercept must be
    excluded. The modal count favors smaller counts on frequency ties;
    median absolute coefficients determine the final zero-based selection.
    All-zero rows contribute zero; other constant magnitudes are rejected.
    Exact clustering is deterministic. Optional kmeans uses NumPy Lloyd
    iteration and does not reproduce R's RNG or default k-means algorithm.
    """
    draws = matrix(draws, "draws", min_cols=2)
    feature_names = names(feature_names, draws.shape[1])
    clustering = _choice(clustering, ("exact", "kmeans"), "clustering")
    generator = rng(random_state)
    magnitudes = np.abs(draws)
    zero = np.all(magnitudes == 0, axis=1)
    constant = np.all(magnitudes == magnitudes[:, :1], axis=1)
    if np.any(constant & ~zero):
        raise ValueError(
            "Nonzero constant-magnitude draws have no defined two-means split."
        )
    counts = np.zeros(len(draws), dtype=int)
    for i in np.flatnonzero(~zero):
        labels, _ = split_two_means(magnitudes[i], clustering, generator)
        counts[i] = np.bincount(labels, minlength=2).min()
    medians = _medians(magnitudes)
    count = _mode(counts)
    return SelectionResult(
        _rank(medians, count),
        "2M",
        medians,
        counts,
        feature_names,
        {
            "clustering": clustering,
            "zero_draws": np.flatnonzero(zero),
            "frequencies": _frequencies(counts),
            "index_base": 0,
        },
    )


def _noise_count(magnitudes, threshold, clustering, small_set, generator):
    if len(magnitudes) == 2:
        return 1
    labels, centers = split_two_means(magnitudes, clustering, generator)
    saved = int(np.count_nonzero(labels == 0))
    # Save D before splitting it again. On stopping, use saved D rather than
    # the lower cluster of the last split (the historical stopping bug).
    while centers[1] - centers[0] > threshold:
        magnitudes = magnitudes[labels == 0]
        saved = len(magnitudes)
        if (
            saved <= 1
            or (small_set == "legacy" and saved <= 2)
            or np.all(magnitudes == magnitudes[0])
        ):
            break
        labels, centers = split_two_means(magnitudes, clustering, generator)
    return saved


def s2m(
    draws,
    thresholds,
    *,
    clustering="exact",
    small_set="legacy",
    zero_draws="error",
    random_state=None,
    feature_names=None,
):
    """Count signals by recursively splitting the lower-mean noise candidate.

    Rows are posterior draws, columns exclude the intercept. Thresholds are
    finite nonnegative center gaps on the coefficient scale. Recursion uses
    the strict gap > threshold rule and returns the saved noise-set size.
    legacy retains recursive pairs; refine splits them before rechecking the
    gap. Both retain singletons and constant candidates. Initial nonzero
    constant magnitudes are undefined. zero_draws='empty' explicitly assigns
    zero signals to all-zero rows, including them in modal denominators.

    Counts use the smaller mode on frequency ties. Result selections rank
    median absolute coefficients, with larger zero-based indices winning
    ties. Exact clustering is deterministic; kmeans uses NumPy Lloyd rather
    than R's RNG and default k-means algorithm. These are draw summaries,
    not inclusion probabilities or error-control guarantees.
    """
    draws = matrix(draws, "draws", min_cols=2)
    thresholds = vector(thresholds, "thresholds")
    if np.any(thresholds < 0):
        raise ValueError("thresholds must not contain negative values.")
    feature_names = names(feature_names, draws.shape[1])
    clustering = _choice(clustering, ("exact", "kmeans"), "clustering")
    small_set = _choice(small_set, ("legacy", "refine"), "small_set")
    zero_draws = _choice(zero_draws, ("error", "empty"), "zero_draws")
    generator = rng(random_state)
    magnitudes = np.abs(draws)
    zero = np.all(magnitudes == 0, axis=1)
    constant = np.all(magnitudes == magnitudes[:, :1], axis=1)
    if np.any(constant & ~(zero & (zero_draws == "empty"))):
        raise ValueError(
            "Each draw must have two distinct absolute coefficients; "
            "only zero_draws='empty' permits an all-zero row."
        )
    draw_counts = np.zeros((len(thresholds), len(draws)), dtype=int)
    for j, threshold in enumerate(thresholds):
        for i in np.flatnonzero(~zero):
            draw_counts[j, i] = draws.shape[1] - _noise_count(
                magnitudes[i], threshold, clustering, small_set, generator
            )
    counts = np.array([_mode(row) for row in draw_counts], dtype=int)
    return TuningResult(
        thresholds.copy(),
        counts,
        draw_counts,
        _medians(magnitudes),
        feature_names,
        {
            "clustering": clustering,
            "small_set": small_set,
            "zero_draws_policy": zero_draws,
            "zero_draws": np.flatnonzero(zero),
            "index_base": 0,
        },
    )


def spike_slab(draws, *, feature_names=None):
    """Summarize exact-zero spike-and-slab draws without fitting a model.

    Any nonzero coefficient counts as included, without a tolerance. The
    median probability model (MPM) uses inclusion frequency >= 0.5. The
    empirical highest posterior probability model (HPPM) is the most frequent
    visited joint pattern; it does not establish the global posterior mode.
    Visited models and HPPM ties follow first-visit order. All indices are
    zero-based. Exclude the intercept before supplying draws.
    """
    draws = matrix(draws, "draws")
    feature_names = names(feature_names, draws.shape[1])
    included = draws != 0
    lookup = {}
    visited, counts, first = [], [], []
    for i, row in enumerate(included):
        key = row.tobytes()
        index = lookup.get(key)
        if index is None:
            index = len(visited)
            lookup[key] = index
            visited.append(row.copy())
            counts.append(0)
            first.append(i)
        counts[index] += 1
    visited = np.asarray(visited, dtype=bool)
    counts = np.asarray(counts, dtype=int)
    ties = np.flatnonzero(counts == counts.max())
    probability = included.mean(axis=0)
    return SpikeSlabResult(
        probability,
        np.flatnonzero(probability >= 0.5),
        np.flatnonzero(visited[ties[0]]),
        visited,
        counts,
        ties,
        feature_names,
        {
            "first_draw": np.asarray(first, dtype=int),
            "model_proportions": counts / len(draws),
            "index_base": 0,
        },
    )
