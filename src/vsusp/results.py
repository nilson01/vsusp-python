"""Results for posterior selection and Gaussian regression."""

from dataclasses import dataclass, field
import numpy as np
from ._validation import matrix, scalar, integer, rng, names, _float64


def _statistics(values, interval, *, sd=False):
    """Scale each coordinate before reductions; reject unrepresentable ranges."""
    scale = np.max(np.abs(values), axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scaled = values / scale
        if np.any((values != 0) & (scaled == 0)):
            raise ValueError(
                "Posterior range is too wide for stable float64 summaries."
            )
        q = (1 - interval) / 2
        mean = scaled.mean(0) * scale
        lo, hi = np.quantile(scaled, [q, 1 - q], axis=0) * scale
        std = scaled.std(0, ddof=1) * scale if sd and len(values) > 1 else None
    fields = [mean, lo, hi] + ([] if std is None else [std])
    if any(not np.isfinite(x).all() for x in fields):
        raise ValueError("Posterior summary exceeded finite numeric range.")
    return mean, lo, hi, std


def _rank(medians, count):
    return np.lexsort((-np.arange(len(medians)), -np.asarray(medians)))[:count]


@dataclass
class SelectionResult:
    selected: np.ndarray
    method: str
    median_abs: np.ndarray
    draw_counts: np.ndarray
    feature_names: tuple
    metadata: dict = field(default_factory=dict)

    @property
    def count(self):
        return len(self.selected)

    @property
    def selected_names(self):
        return tuple(self.feature_names[j] for j in self.selected)

    def summary(self):
        return f"{self.method}: {self.count} selected\n" + ", ".join(
            self.selected_names
        )


@dataclass
class TuningResult:
    thresholds: np.ndarray
    counts: np.ndarray
    draw_counts: np.ndarray
    median_abs: np.ndarray
    feature_names: tuple
    metadata: dict = field(default_factory=dict)

    @property
    def mode_proportion(self):
        return np.mean(self.draw_counts == self.counts[:, None], axis=1)

    @property
    def selected_sets(self):
        return tuple(_rank(self.median_abs, int(h)) for h in self.counts)

    @property
    def jaccard_previous(self):
        out = np.full(len(self.counts), np.nan)
        for i in range(1, len(out)):
            a, b = map(set, (self.selected_sets[i - 1], self.selected_sets[i]))
            out[i] = len(a & b) / len(a | b) if a | b else 1.0
        return out

    def select(self, index):
        i = integer(index, "index")
        if i >= len(self.thresholds):
            raise IndexError("threshold index is out of range.")
        return SelectionResult(
            self.selected_sets[i],
            "S2M",
            self.median_abs.copy(),
            self.draw_counts[i].copy(),
            self.feature_names,
            {**self.metadata, "threshold": float(self.thresholds[i])},
        )

    def summary(self):
        rows = ["threshold  count  modal frequency"]
        rows += [
            f"{b:9.4g}  {h:5d}  {m:.3f}"
            for b, h, m in zip(self.thresholds, self.counts, self.mode_proportion)
        ]
        return "\n".join(rows)


@dataclass
class SpikeSlabResult:
    inclusion_probability: np.ndarray
    mpm: np.ndarray
    hppm: np.ndarray
    visited_models: np.ndarray
    model_counts: np.ndarray
    hppm_ties: np.ndarray
    feature_names: tuple
    metadata: dict = field(default_factory=dict)

    def summary(self):
        rows = ["variable  inclusion frequency"]
        rows += [
            f"{n}  {v:.4f}"
            for n, v in zip(self.feature_names, self.inclusion_probability)
        ]
        rows += [
            f"MPM: {len(self.mpm)} selected; visited HPPM: {len(self.hppm)} selected"
        ]
        return "\n".join(rows)


@dataclass
class SparseResult:
    coefficients: np.ndarray
    selected: np.ndarray
    method: str
    feature_names: tuple
    converged: bool = True
    iterations: int = 0
    objective: float = float("nan")
    diagnostics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def predict(self, X, *, intercept=0.0):
        X = matrix(X)
        if X.shape[1] != len(self.coefficients):
            raise ValueError(
                "X must have one column per coefficient, in the fitted order."
            )
        values = X @ self.coefficients + scalar(intercept, "intercept")
        if not np.isfinite(values).all():
            raise ValueError("Prediction arithmetic exceeded finite numeric range.")
        return values

    def summary(self):
        rows = [
            f"{self.method}: {len(self.selected)} selected; converged={self.converged}"
        ]
        rows += [
            f"{self.feature_names[j]}  {self.coefficients[j]:.6g}"
            for j in self.selected
        ]
        return "\n".join(rows)


@dataclass
class PredictionResult:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    interval: float
    kind: str


@dataclass
class GaussianFit:
    coefficient_draws: np.ndarray
    intercept_draws: np.ndarray
    sigma2_draws: np.ndarray
    prior: str
    feature_names: tuple
    extra: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        raw = np.asarray(self.coefficient_draws)
        if raw.ndim != 3 or min(raw.shape) < 1 or raw.dtype.kind not in "iuf":
            raise ValueError(
                "coefficient_draws must have finite chain, draw and coefficient axes."
            )
        self.coefficient_draws = _float64(raw, "coefficient_draws")
        self.intercept_draws = matrix(self.intercept_draws, "intercept_draws")
        self.sigma2_draws = matrix(self.sigma2_draws, "sigma2_draws")
        if (
            self.intercept_draws.shape != raw.shape[:2]
            or self.sigma2_draws.shape != raw.shape[:2]
        ):
            raise ValueError(
                "All posterior arrays must have matching chain and draw axes."
            )
        if np.any(self.sigma2_draws <= 0):
            raise ValueError("sigma2_draws must be positive.")
        self.feature_names = names(self.feature_names, raw.shape[2])

    def pooled_draws(self):
        """Combine chains for selection, keeping this fit's chain array intact."""
        self._validate()
        return self.coefficient_draws.reshape(
            -1, self.coefficient_draws.shape[-1]
        ).copy()

    def summary(self, interval=0.95):
        interval = scalar(interval, "interval")
        if not 0 < interval < 1:
            raise ValueError("interval must be strictly between zero and one.")
        d = self.pooled_draws()
        mean, lo, hi, std = _statistics(d, interval, sd=True)
        rows = [
            f"Gaussian {self.prior}; posterior summaries ({interval:.0%} intervals)",
            "variable  mean  sd  lower  upper",
        ]
        for j, n in enumerate(self.feature_names):
            text_sd = "unavailable" if std is None else f"{std[j]:.5g}"
            rows.append(f"{n}  {mean[j]:.5g}  {text_sd}  {lo[j]:.5g}  {hi[j]:.5g}")
        if std is None:
            rows.append("SD unavailable: only one pooled draw.")
        return "\n".join(rows)

    def predict(self, X, *, interval=0.95, include_noise=False, random_state=None):
        """Pointwise posterior intervals for the mean or a new observation."""
        X = matrix(X)
        if X.shape[1] != len(self.feature_names):
            raise ValueError("X must preserve the fitted predictor count and order.")
        interval = scalar(interval, "interval")
        if not 0 < interval < 1:
            raise ValueError("interval must be strictly between zero and one.")
        if not isinstance(include_noise, (bool, np.bool_)):
            raise ValueError("include_noise must be boolean.")
        values = self.pooled_draws() @ X.T + self.intercept_draws.reshape(-1, 1)
        if not np.isfinite(values).all():
            raise ValueError("Prediction arithmetic exceeded finite numeric range.")
        mean = _statistics(values, interval)[0]
        if include_noise:
            values += rng(random_state).normal(size=values.shape) * np.sqrt(
                self.sigma2_draws.reshape(-1, 1)
            )
        if not np.isfinite(values).all():
            raise ValueError("Prediction arithmetic exceeded finite numeric range.")
        _, lo, hi, _ = _statistics(values, interval)
        return PredictionResult(
            mean,
            lo,
            hi,
            interval,
            "new observation" if include_noise else "conditional mean",
        )
