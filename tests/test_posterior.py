"""Label-preserving legacy InferenceData imports on fixed supplied draws."""

import numpy as np
import pytest
import xarray as xr
from arviz.data.inference_data import InferenceData
from numpy.testing import assert_array_equal
from vsusp import PosteriorDraws, from_inferencedata, select


def example():
    values = np.arange(24, dtype=float).reshape(2, 4, 3)
    dataset = xr.Dataset(
        {'beta': (('feature', 'draw', 'chain'), values.transpose(2, 1, 0)),
         'intercept': (('chain', 'draw'), np.ones((2, 4)))},
        coords={'chain': [7, 2], 'draw': [40, 30, 20, 10],
                'feature': ['a', 'b', 'Intercept']},
    )
    return InferenceData(posterior=dataset, warmup_posterior=dataset), values


def test_selected_columns_preserve_chain_draw_order_and_pool_explicitly():
    data, values = example()
    result = from_inferencedata(data, var_names=['beta', 'intercept'],
                               coords={'feature': ['b', 'a']}, discard=1)
    expected = np.concatenate([values[:, 1:, [1, 0]], np.ones((2, 3, 1))], axis=2)
    assert_array_equal(result.as_array(), expected)
    assert result.chain_labels == (7, 2)
    assert result.draw_labels == (30, 20, 10)
    assert result.feature_names == ("beta['b']", "beta['a']", 'intercept')
    assert result.source_variables == ('beta', 'beta', 'intercept')
    assert_array_equal(result.pooled(), expected.reshape(6, 3))
    assert select(result.pooled(), 1, feature_names=result.feature_names).selected_names == ("beta['b']",)
    copy = result.as_array()
    copy[:] = 0
    assert np.any(result.values != 0)
    assert_array_equal(data.posterior.beta.values, values.transpose(2, 1, 0))


@pytest.mark.parametrize('options', [
    {'var_names': 'beta'}, {'var_names': []}, {'var_names': ['beta', 'beta']},
    {'var_names': ['absent']}, {'var_names': ['beta'], 'discard': True},
    {'var_names': ['beta'], 'discard': 4},
    {'var_names': ['beta'], 'coords': {'feature': ['absent']}},
    {'var_names': ['beta'], 'coords': {'feature': ['a', 'a']}},
    {'var_names': ['beta'], 'coords': {'draw': [40]}},
])
def test_invalid_selection_fails(options):
    data, _ = example()
    with pytest.raises(ValueError):
        from_inferencedata(data, **options)


def test_missing_nonfinite_duplicate_and_unsupported_axes_fail():
    data, _ = example()
    with pytest.raises(TypeError):
        from_inferencedata({'posterior': data.posterior}, var_names=['beta'])
    with pytest.raises(ValueError, match='posterior group'):
        from_inferencedata(InferenceData(), var_names=['beta'])
    for dataset in [data.posterior.assign_coords(chain=[1, 1]),
                    data.posterior.assign_coords(feature=['a', 'a', 'b'])]:
        with pytest.raises(ValueError, match='unique'):
            from_inferencedata(InferenceData(posterior=dataset), var_names=['beta'])
    data.posterior.beta.values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        from_inferencedata(data, var_names=['beta'])
    dataset = xr.Dataset({'matrix': (('chain', 'draw', 'a', 'b'), np.ones((1, 2, 2, 2)))},
                         coords={'chain': [0], 'draw': [0, 1], 'a': [0, 1], 'b': [0, 1]})
    with pytest.raises(ValueError, match='at most one'):
        from_inferencedata(InferenceData(posterior=dataset), var_names=['matrix'])


def test_container_validates_axes_names_and_finite_values():
    with pytest.raises(ValueError):
        PosteriorDraws(np.ones((1, 2, 1)), (0,), (0, 1), ('beta', 'extra'), ('beta',))
