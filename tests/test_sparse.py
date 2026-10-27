import numpy as np
import pytest
from vsusp import savs, dss


def test_savs_analytic_and_equality():
    np.testing.assert_allclose(savs(np.eye(2), [2, 0.1]).coefficients, [1.75, 0])
    assert savs(np.eye(2), [1, -1]).selected.size == 0
    assert savs(np.eye(2), [0, 0]).selected.size == 0


def test_dss_orthogonal_soft_threshold_oracle():
    fit = dss(np.eye(2), [2, 0.1], 0.2)
    np.testing.assert_allclose(fit.coefficients, [1.9, 0], atol=1e-12)
    assert fit.objective == pytest.approx(0.2)
    assert fit.converged and fit.diagnostics["kkt_error"] < 1e-8
    np.testing.assert_allclose(fit.predict(np.eye(2)), [1.9, 0])
    assert "DSS" in fit.summary()


def test_dss_zero_penalty_zero_mean_and_empty_selection():
    np.testing.assert_equal(dss(np.eye(2), [0.0, 2], 0).coefficients, [0, 2])
    assert dss(np.eye(2), [0.0, 2], 10).selected.size == 0
    assert dss(np.eye(2), [0.0, 0], 1).converged


def test_dss_rescaling_preserves_prediction():
    X = np.array([[1.0, 0], [0, 2], [1, 1]])
    base = dss(X, [2, 0.5], 0.1)
    scale = np.array([2.0, 0.5])
    other = dss(X * scale, np.array([2, 0.5]) / scale, 0.1)
    np.testing.assert_allclose(
        base.predict(X), other.predict(X * scale), rtol=1e-7, atol=1e-8
    )


def test_dss_dependent_predictors_and_nonconvergence():
    X = np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
    with pytest.warns(RuntimeWarning):
        out = dss(X, [1.0, 1], 0.1, tol=1e-15, max_iter=1)
    assert not out.converged


@pytest.mark.parametrize("fun", [savs, lambda X, beta, **kw: dss(X, beta, 0.1, **kw)])
def test_sparse_invalid_inputs(fun):
    for X, b in [
        (np.zeros((2, 2)), [1, 1]),
        (np.eye(2), [1]),
        (np.eye(2), [np.nan, 1]),
    ]:
        with pytest.raises(ValueError):
            fun(X, b)
    with pytest.raises(ValueError):
        fun(np.eye(2), [1, 1], feature_names=["a", "a"])


def test_sparse_controls_and_numeric_range():
    for p in [-1, np.nan, True]:
        with pytest.raises(ValueError):
            dss(np.eye(2), [1, 1], p)
    with pytest.raises(ValueError):
        dss(np.eye(2), [1, 1], 1, tol=0)
    with pytest.raises(ValueError):
        dss(np.eye(2), [1, 1], 1, max_iter=1.5)
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError):
            dss(np.eye(2) * 1e200, [1.0, 1], 0.1)
