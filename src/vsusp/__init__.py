"""Variable selection using shrinkage-prior posterior draws."""

from .selection import select, s2m, two_means, spike_slab
from .sparse import savs, dss
from .gaussian import fit_horseshoe, fit_spike_slab, fit_ridge
from .diagnostics import diagnostics, DiagnosticResult
from .results import (
    SelectionResult,
    TuningResult,
    SpikeSlabResult,
    SparseResult,
    GaussianFit,
    PredictionResult,
)
from .plots import (
    plot_tuning,
    plot_coefficients,
    plot_inclusion,
    plot_trace,
    plot_autocorrelation,
)

__version__ = "0.1.0"
__all__ = [
    "select",
    "s2m",
    "two_means",
    "spike_slab",
    "savs",
    "dss",
    "fit_horseshoe",
    "fit_spike_slab",
    "fit_ridge",
    "diagnostics",
    "plot_tuning",
    "plot_coefficients",
    "plot_inclusion",
    "plot_trace",
    "plot_autocorrelation",
    "SelectionResult",
    "TuningResult",
    "SpikeSlabResult",
    "SparseResult",
    "GaussianFit",
    "PredictionResult",
    "DiagnosticResult",
]
