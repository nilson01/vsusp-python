"""Explicit conversion of legacy ArviZ posterior groups to coefficient draws."""

from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np


def _labels(values, name):
    try:
        labels = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of labels.") from exc
    if not labels:
        raise ValueError(f"{name} must not be empty.")
    try:
        if len(set(labels)) != len(labels) or any(v is None or v != v or (isinstance(v, (float, np.floating)) and not np.isfinite(v))
                                                     for v in labels):
            raise ValueError(f"{name} must be unique and nonmissing.")
    except TypeError as exc:
        raise ValueError(f"{name} must contain scalar labels.") from exc
    return labels


@dataclass(frozen=True)
class PosteriorDraws:
    """Coefficient draws with explicit chain, retained-draw and column labels.

    Use ``as_array()`` for diagnostics and ``pooled()`` for selection. Importing
    draws does not fit a model or assess its sampling or statistical assumptions.
    """

    values: np.ndarray
    chain_labels: tuple
    draw_labels: tuple
    feature_names: tuple
    source_variables: tuple

    def __post_init__(self):
        raw = np.asarray(self.values)
        if raw.ndim != 3 or min(raw.shape) < 1 or raw.dtype.kind not in 'iuf':
            raise ValueError("values must be real (chain, draw, coefficient) arrays.")
        with np.errstate(over='ignore', invalid='ignore'):
            values = np.array(raw, dtype=float, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("posterior draws must be finite.")
        for name, size in zip(('chain_labels', 'draw_labels', 'feature_names'), values.shape):
            labels = _labels(getattr(self, name), name)
            if len(labels) != size:
                raise ValueError(f"{name} must match its array axis.")
            object.__setattr__(self, name, labels)
        if any(not isinstance(n, str) or not n or not n.isprintable()
               for n in self.feature_names):
            raise ValueError("feature_names must be printable nonempty strings.")
        provenance = tuple(self.source_variables)
        if len(provenance) != values.shape[2] or any(
            not isinstance(n, str) or not n for n in provenance
        ):
            raise ValueError("source_variables must name each coefficient's variable.")
        object.__setattr__(self, 'source_variables', provenance)
        values.setflags(write=False)
        object.__setattr__(self, 'values', values)

    def as_array(self):
        """Return a copy with separate chain, draw and coefficient axes."""
        return self.values.copy()

    def pooled(self):
        """Return a copy with chains pooled in chain-major order for selection."""
        return self.values.reshape(-1, len(self.feature_names)).copy()


def from_inferencedata(idata, *, var_names, coords=None, discard=0):
    """Import explicitly selected variables from legacy ArviZ InferenceData.

    Install ``vsusp[posterior]`` (ArviZ >=0.18,<1). Only ``posterior`` is read;
    warmup groups are ignored. Each variable needs named chain and draw axes
    and at most one coefficient axis. ``coords`` maps coefficient dimensions
    to ordered labels to retain; omitted dimensions retain all their labels.
    Intercepts are included only when selected, without automatic name rules.
    ``discard`` removes that many leading posterior draws from every chain.
    DataTree containers from ArviZ 1.x are not supported by this adapter.
    """
    try:
        from arviz.data.inference_data import InferenceData
    except ImportError as exc:
        raise ImportError('Install legacy posterior support: pip install "vsusp[posterior]"') from exc
    if not isinstance(idata, InferenceData):
        raise TypeError("idata must be legacy ArviZ InferenceData, not a DataTree or mapping.")
    if not hasattr(idata, 'posterior'):
        raise ValueError("InferenceData must contain a posterior group.")
    if isinstance(var_names, str):
        raise ValueError("var_names must be an explicit sequence of variable names.")
    try:
        variables = _labels(var_names, 'var_names')
    except TypeError as exc:
        raise ValueError("var_names must be an explicit sequence of variable names.") from exc
    if any(not isinstance(n, str) or not n for n in variables):
        raise ValueError("var_names must contain nonempty strings.")
    if isinstance(discard, (bool, np.bool_)) or not isinstance(discard, (int, np.integer)) or discard < 0:
        raise ValueError("discard must be a nonnegative integer.")
    if coords is None:
        coords = {}
    if not isinstance(coords, Mapping) or any(not isinstance(d, str) for d in coords):
        raise ValueError("coords must map coefficient dimensions to label sequences.")
    dataset = idata.posterior
    for dim in ('chain', 'draw'):
        if dim not in dataset.coords or dataset.coords[dim].dims != (dim,):
            raise ValueError(f"posterior must have a one-dimensional {dim} coordinate.")
    chains = _labels(dataset.coords['chain'].values.tolist(), 'chain coordinates')
    draws = _labels(dataset.coords['draw'].values.tolist(), 'draw coordinates')
    if discard >= len(draws):
        raise ValueError("discard must leave at least one posterior draw.")
    parts, names, provenance, used = [], [], [], set()
    for name in variables:
        if name not in dataset.data_vars:
            raise ValueError(f"posterior variable not found: {name}")
        value = dataset[name]
        extra = [d for d in value.dims if d not in ('chain', 'draw')]
        if 'chain' not in value.dims or 'draw' not in value.dims or len(extra) > 1:
            raise ValueError(f"{name} needs chain/draw axes and at most one coefficient axis.")
        if extra:
            dim = extra[0]
            if dim not in value.coords or value.coords[dim].dims != (dim,):
                raise ValueError(f"{dim} needs explicit one-dimensional coefficient coordinates.")
            _labels(value.coords[dim].values.tolist(), f'{dim} coordinates')
            if dim in coords:
                if isinstance(coords[dim], str):
                    raise ValueError("coords selections must be sequences, not strings.")
                selected = _labels(coords[dim], f'{dim} selection')
                try:
                    value = value.sel({dim: list(selected)})
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"unknown coefficient coordinate in {dim} selection.") from exc
                used.add(dim)
            labels = value.coords[dim].values.tolist()
            names.extend(f'{name}[{label!r}]' for label in labels)
            raw = value.transpose('chain', 'draw', dim).values[:, discard:, :]
        else:
            names.append(name)
            raw = value.transpose('chain', 'draw').values[:, discard:, None]
        # Validate before concatenation so mixed string/numeric variables cannot coerce.
        if raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
            raise ValueError(f"{name} must contain finite real posterior draws.")
        parts.append(raw)
        provenance.extend([name] * raw.shape[-1])
    if set(coords) != used:
        raise ValueError("coords contains a dimension absent from the selected coefficient variables.")
    return PosteriorDraws(np.concatenate(parts, axis=2), chains, draws[discard:],
                          tuple(names), tuple(provenance))
