import itertools
import numpy as np
import pytest
from vsusp import select, s2m, two_means, spike_slab
from vsusp._clustering import exact_two_means

B = np.array(
    [
        [0.01, 0.02, 0.03, 0.3, 0.31, 10, 11],
        [-0.01, -0.02, -0.03, -0.3, -0.31, -10, -11],
    ]
)


def test_saved_noise_set_regression_and_ranking():
    out = s2m(B, [0.1, 0.5, 100.0])
    np.testing.assert_array_equal(out.counts, [4, 2, 2])
    np.testing.assert_array_equal(out.select(1).selected, [6, 5])
    assert out.select(1).selected_names == ("x7", "x6")
    assert np.all(out.mode_proportion == 1)
    assert "threshold" in out.summary()


def test_recursive_pair_conventions():
    b = np.array([[0, 0.125, 10.0]])
    assert s2m(b, [0.1], small_set="legacy").counts[0] == 1
    assert s2m(b, [0.1], small_set="refine").counts[0] == 2
    assert s2m([[0.0, 4.0]], [0, 10]).counts.tolist() == [1, 1]


def test_two_means_smaller_cluster_not_upper_cluster():
    out = two_means([[0.0, 0.1, 10, 11, 12]])
    assert out.count == 2
    np.testing.assert_array_equal(out.selected, [4, 3])


def test_zero_draw_boundaries_and_count_ties():
    b = [[0.0, 0, 0], [0, 0, 0], [0, 0.1, 10]]
    with pytest.raises(ValueError):
        s2m(b, [1])
    assert s2m(b, [1], zero_draws="empty").counts[0] == 0
    assert two_means(b).count == 0
    np.testing.assert_array_equal(select(B, 0).selected, [])
    for fun in [lambda: s2m([[2.0, -2, 2]], [1]), lambda: two_means([[2.0, -2, 2]])]:
        with pytest.raises(ValueError):
            fun()


def test_spike_joint_model_differs_from_marginal_model():
    b = np.array([[1.0, 0, 0], [2, 0, 0], [0, -1, 0], [0, -2, 0]])
    out = spike_slab(b)
    np.testing.assert_allclose(out.inclusion_probability, [0.5, 0.5, 0])
    np.testing.assert_array_equal(out.mpm, [0, 1])
    np.testing.assert_array_equal(out.hppm, [0])
    np.testing.assert_array_equal(out.hppm_ties, [0, 1])
    assert not out.visited_models.all(1).any()
    assert len(spike_slab(np.zeros((2, 3))).mpm) == 0
    assert spike_slab([[0.0, 1e-300]]).mpm.tolist() == [1]


def test_median_tie_does_not_claim_inclusion():
    b = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    assert two_means(b).selected.tolist() == [3]
    assert spike_slab(b).inclusion_probability[3] == 0


@pytest.mark.parametrize(
    "values", [[0.0, 0.1, 1, 3, 8], [0.0, 0, 1, 1, 2], [0.0, 1, 2, 3]]
)
def test_exact_cluster_loss_matches_all_binary_partitions(values):
    x = np.array(values)
    labels, _ = exact_two_means(x)
    loss = sum(np.sum((x[labels == k] - x[labels == k].mean()) ** 2) for k in [0, 1])
    losses = []
    for bits in itertools.product([False, True], repeat=len(x)):
        mask = np.array(bits)
        if mask.any() and (~mask).any():
            losses.append(
                np.sum((x[mask] - x[mask].mean()) ** 2)
                + np.sum((x[~mask] - x[~mask].mean()) ** 2)
            )
    assert loss == pytest.approx(min(losses), abs=1e-13)


def test_lloyd_replay_and_no_global_rng_change():
    np.random.seed(72)
    saved = np.random.get_state()
    a = s2m(B, [0.1, 0.5], clustering="kmeans", random_state=91)
    b = s2m(B, [0.1, 0.5], clustering="kmeans", random_state=91)
    np.testing.assert_array_equal(a.draw_counts, b.draw_counts)
    after = np.random.get_state()
    assert (
        saved[0] == after[0]
        and np.array_equal(saved[1], after[1])
        and saved[2:] == after[2:]
    )


@pytest.mark.parametrize(
    "bad",
    [
        [1.0, 2.0],
        [[True, False]],
        [[1j, 2j]],
        [[np.nan, 1]],
        [[np.inf, 1]],
        [["1", "2"]],
        np.empty((0, 2)),
    ],
)
def test_draw_inputs_reject_invalid_values(bad):
    for f in [
        lambda: select(bad, 1),
        lambda: two_means(bad),
        lambda: s2m(bad, [1]),
        lambda: spike_slab(bad),
    ]:
        with pytest.raises(ValueError):
            f()


def test_lossy_integer_conversion_is_rejected():
    with pytest.raises(ValueError, match="precision"):
        select(np.array([[9007199254740993, 9007199254740992]], dtype=np.int64), 1)
    assert select([[2**54, 0]], 1).selected.tolist() == [0]


@pytest.mark.parametrize(
    "kw", [{"count": True}, {"count": 1.5}, {"count": -1}, {"count": 8}]
)
def test_count_control(kw):
    with pytest.raises(ValueError):
        select(B, **kw)


def test_names_and_threshold_controls():
    with pytest.raises(ValueError):
        s2m(B, [-1])
    with pytest.raises(ValueError):
        s2m(B, [])
    with pytest.raises(ValueError):
        select(B, 1, feature_names=["a"] * 7)
    with pytest.raises(IndexError):
        s2m(B, [1]).select(1)
