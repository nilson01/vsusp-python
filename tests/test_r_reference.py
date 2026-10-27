"""Pinned repaired installed-R values, not released-CRAN equivalence."""

from pathlib import Path
import numpy as np
from vsusp import s2m, two_means, spike_slab, savs, dss

P = Path(__file__).parent / "data" / "r_reference"


def read(name):
    return np.loadtxt(P / name, delimiter=",", ndmin=1)


def test_installed_r_s2m_fixed_draws():
    expected = read("s2m.csv")
    np.testing.assert_equal(
        s2m(read("draws.csv"), expected[:, 0]).counts, expected[:, 1]
    )
    pairs = [
        s2m([[0.0, 0.125, 10]], [0.1], small_set=method).counts[0]
        for method in ["legacy", "refine"]
    ]
    np.testing.assert_equal(pairs, read("pairs.csv"))


def test_installed_r_two_means_and_spike_summaries():
    np.testing.assert_equal(
        two_means(read("cardinality.csv").reshape(1, -1)).selected,
        read("two-means-selected.csv"),
    )
    ss = spike_slab(read("spike.csv"))
    np.testing.assert_equal(ss.inclusion_probability, read("inclusion.csv"))
    np.testing.assert_equal(ss.mpm, read("mpm.csv"))
    np.testing.assert_equal(ss.hppm, read("hppm.csv"))


def test_installed_r_sparse_coefficients_and_objective():
    X, beta = read("X.csv"), read("beta.csv")
    np.testing.assert_allclose(savs(X, beta).coefficients, read("savs.csv"), atol=1e-12)
    fit = dss(X, beta, 0.1)
    np.testing.assert_allclose(fit.coefficients, read("dss.csv"), atol=1e-10)
    np.testing.assert_allclose(
        fit.objective, read("dss-diagnostics.csv")[0], atol=1e-12
    )
