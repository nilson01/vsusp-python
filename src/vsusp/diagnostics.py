"""Chain-aware diagnostics delegated to ArviZ."""

from dataclasses import dataclass
import numpy as np
from ._validation import names, _float64
from .results import GaussianFit


def chain_array(value):
    """Require explicit (chain, retained draw, coefficient) axes."""
    if isinstance(value, GaussianFit):
        value._validate()
    raw = (
        value.coefficient_draws if isinstance(value, GaussianFit) else np.asarray(value)
    )
    if raw.ndim != 3 or min(raw.shape) < 1 or raw.dtype.kind not in "iuf":
        raise ValueError("draws must have explicit chain, draw and coefficient axes.")
    return _float64(raw, "draws")


@dataclass
class DiagnosticResult:
    feature_names: tuple
    rhat: np.ndarray
    ess_bulk: np.ndarray
    ess_tail: np.ndarray
    mcse_mean: np.ndarray
    flags: tuple
    backend: str

    def summary(self):
        rows = ["variable  rank R-hat  bulk ESS  tail ESS  mean MCSE  status"]
        rows += [
            f"{n}  {r:.4g}  {b:.5g}  {t:.5g}  {m:.4g}  {f}"
            for n, r, b, t, m, f in zip(
                self.feature_names,
                self.rhat,
                self.ess_bulk,
                self.ess_tail,
                self.mcse_mean,
                self.flags,
            )
        ]
        return "\n".join(rows)


def diagnostics(draws, *, feature_names=None):
    """Compute rank R-hat, bulk/tail ESS and mean MCSE using ArviZ.

    Install ``vsusp[diagnostics]``. Arrays must retain separate chain and draw
    axes; a pooled matrix cannot recover chain identity. A GaussianFit also
    reports its intercept and residual variance. Single-chain R-hat and
    constant-variable diagnostics are unavailable, not evidence of convergence.
    These diagnostics do not establish that chains explored every posterior mode.
    """
    arr = chain_array(draws)
    if arr.shape[1] < 4:
        raise ValueError("Diagnostics require at least four retained draws per chain.")
    if isinstance(draws, GaussianFit):
        if feature_names is not None:
            raise ValueError("GaussianFit already supplies coefficient names.")
        labels = tuple(f"beta[{n}]" for n in draws.feature_names) + (
            "intercept",
            "sigma2",
        )
        arr = np.concatenate(
            [arr, draws.intercept_draws[:, :, None], draws.sigma2_draws[:, :, None]],
            axis=2,
        )
    else:
        labels = names(feature_names, arr.shape[2])
    try:
        import arviz_stats as backend
    except ImportError:
        try:
            import arviz as backend
        except ImportError as exc:
            raise ImportError(
                'Install diagnostics with: pip install "vsusp[diagnostics]"'
            ) from exc
    rhat, bulk, tail, mcse = (np.full(arr.shape[2], np.nan) for _ in range(4))
    flags = []
    for j in range(arr.shape[2]):
        a = arr[:, :, j]
        if np.all(a == a[0, 0]):
            flags.append("constant; unavailable")
            continue
        flag = "inspect diagnostics"
        if arr.shape[0] >= 2:
            rhat[j] = float(backend.rhat(a, method="rank"))
        else:
            flag = "one chain; R-hat unavailable"
        bulk[j] = float(backend.ess(a, method="bulk"))
        tail[j] = float(backend.ess(a, method="tail", prob=(0.05, 0.95)))
        mcse[j] = float(backend.mcse(a, method="mean"))
        if not np.isfinite([bulk[j], tail[j], mcse[j]]).all() or (
            arr.shape[0] >= 2 and not np.isfinite(rhat[j])
        ):
            flag += "; some diagnostics unavailable"
        flags.append(flag)
    return DiagnosticResult(
        labels, rhat, bulk, tail, mcse, tuple(flags), backend.__name__
    )
