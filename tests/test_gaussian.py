import numpy as np
import pytest
from vsusp import fit_ridge, fit_horseshoe, fit_spike_slab, GaussianFit, diagnostics
from vsusp.gaussian import _gaussian_block, _gaussian_factor

X = np.array([[-2.0, 0], [-1, 1], [0, -1], [1, 0], [2, 1], [0, -1]])
y = np.array([-3.8, -2.1, 0.1, 2.2, 3.9, -0.2])


@pytest.mark.parametrize("fitter", [fit_ridge, fit_horseshoe, fit_spike_slab])
def test_fit_contracts_seed_replay_and_prediction(fitter):
    kw = {} if fitter is fit_ridge else {"burnin": 10}
    np.random.seed(81)
    saved = np.random.get_state()
    a = fitter(X, y, draws=30, chains=2, random_state=92, **kw)
    b = fitter(X, y, draws=30, chains=2, random_state=92, **kw)
    np.testing.assert_array_equal(a.coefficient_draws, b.coefficient_draws)
    assert a.coefficient_draws.shape == (2, 30, 2)
    assert np.isfinite(a.intercept_draws).all() and (a.sigma2_draws > 0).all()
    assert not np.array_equal(a.coefficient_draws[0], a.coefficient_draws[1])
    assert len(set(a.metadata["chain_seeds"])) == 2
    after = np.random.get_state()
    assert np.array_equal(saved[1], after[1]) and saved[2:] == after[2:]
    pred = a.predict(X)
    assert pred.mean.shape == (6,) and (pred.upper >= pred.lower).all()
    new = a.predict(X, include_noise=True, random_state=77)
    np.testing.assert_equal(pred.mean, new.mean)
    assert new.kind == "new observation"
    assert "posterior" in a.summary()
    if fitter is fit_spike_slab:
        np.testing.assert_equal(a.extra["included"], a.coefficient_draws != 0)


def test_gaussian_affine_mean_and_covariance_are_exact():
    Xc = X - X.mean(0)
    yc = y - y.mean()
    d = np.array([0.3, 2.0])
    factor, rate = _gaussian_factor(Xc, yc, d)
    sig = 0.7
    p = 2
    n = len(y)

    class Controlled:
        def __init__(self, z):
            self.z = z
            self.count = 0

        def gamma(self, shape):
            return rate / sig

        def normal(self, size):
            v = self.z[self.count : self.count + size]
            self.count += size
            return v

    def draw(z):
        return _gaussian_block(Xc, yc, d, Controlled(z), factor, rate)[0]

    center = draw(np.zeros(p + n))
    Q = Xc.T @ Xc + np.diag(1 / d)
    np.testing.assert_allclose(center, np.linalg.solve(Q, Xc.T @ yc), atol=1e-12)
    transform = np.column_stack([draw(e) - center for e in np.eye(p + n)])
    np.testing.assert_allclose(
        transform @ transform.T, sig * np.linalg.inv(Q), atol=1e-12
    )
    C = np.eye(n) + (Xc * d) @ Xc.T
    assert rate == pytest.approx(1.5 + yc @ np.linalg.solve(C, yc) / 2)


def test_ridge_iid_mean_agrees_with_conjugate_posterior():
    f = fit_ridge(X, y, draws=5000, chains=1, random_state=88)
    Xc = X - X.mean(0)
    yc = y - y.mean()
    Q = Xc.T @ Xc + np.eye(2)
    mu = np.linalg.solve(Q, Xc.T @ yc)
    A = 1.5 + (len(y) - 1) / 2
    C = np.eye(len(y)) + Xc @ Xc.T
    expected_sigma = (1.5 + yc @ np.linalg.solve(C, yc) / 2) / (A - 1)
    covariance = expected_sigma * np.linalg.inv(Q)
    delta = abs(f.pooled_draws().mean(0) - mu)
    assert np.all(delta < 6 * np.sqrt(np.diag(covariance) / 5000))
    assert abs(f.sigma2_draws.mean() - expected_sigma) < 0.08


@pytest.mark.parametrize("fitter", [fit_ridge, fit_horseshoe, fit_spike_slab])
@pytest.mark.parametrize(
    "control", [{"draws": 0}, {"draws": 1.5}, {"chains": True}, {"chains": 0}]
)
def test_invalid_fitting_controls(fitter, control):
    with pytest.raises(ValueError):
        fitter(X, y, **control)


@pytest.mark.parametrize("fitter", [fit_ridge, fit_horseshoe, fit_spike_slab])
def test_invalid_fitting_data(fitter):
    with pytest.raises(ValueError):
        fitter(np.ones((3, 2)), [1, 2, 3])
    with pytest.raises(ValueError):
        fitter(X, y[:-1])
    with pytest.raises(ValueError):
        fitter(X[:1], y[:1])


def test_conditioning_policy_fails_without_altering_scales():
    with pytest.raises((FloatingPointError, np.linalg.LinAlgError)):
        _gaussian_factor(
            np.array([[-1.0], [0], [1]]), np.array([-1.0, 0, 1]), np.array([1e12])
        )


def test_result_validation_and_unavailable_sd():
    f = GaussianFit(
        np.ones((1, 1, 1)), np.zeros((1, 1)), np.ones((1, 1)), "ridge", ("x",)
    )
    assert "SD unavailable" in f.summary()
    with pytest.raises(ValueError):
        GaussianFit(
            np.ones((1, 4, 1)), np.zeros((1, 3)), np.ones((1, 4)), "ridge", ("x",)
        )
    with pytest.raises(ValueError):
        GaussianFit(
            np.ones((1, 4, 1)), np.zeros((1, 4)), -np.ones((1, 4)), "ridge", ("x",)
        )
    f.sigma2_draws[:] = -1
    with pytest.raises(ValueError):
        diagnostics(f)


def test_large_finite_reductions_stay_finite():
    f = GaussianFit(
        np.full((1, 20, 1), 1e307), np.zeros((1, 20)), np.ones((1, 20)), "ridge", ("x",)
    )
    assert np.isfinite(f.predict([[1.0]]).mean).all()
    assert "inf" not in f.summary()
