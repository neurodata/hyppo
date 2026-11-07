import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.metrics import pairwise_distances

from hyppo.independence import Dcorr


pytestmark = pytest.mark.filterwarnings(
    "ignore:The number of replications is low:RuntimeWarning"
)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("initial_auto", [False, True])
@pytest.mark.parametrize("auto", [False, True])
@pytest.mark.parametrize("n", [4, 24])
def test_reuse_after_univariate_test(bias, initial_auto, auto, n):
    initial = np.arange(8, dtype=float).reshape(-1, 1)
    test = Dcorr(bias=bias)
    test.test(initial, np.sin(initial), auto=initial_auto, reps=9, random_state=1)

    values = np.arange(n, dtype=float)
    x = np.column_stack((values, np.cos(values)))
    y = np.column_stack((np.sin(values), values % 5))
    fresh = Dcorr(bias=bias)
    expected = fresh.test(x, y, auto=auto, reps=9, random_state=1)
    actual = test.test(x, y, auto=auto, reps=9, random_state=1)

    assert_allclose(actual, expected)
    if fresh.null_dist is None:
        assert test.null_dist is None
    else:
        assert_allclose(test.null_dist, fresh.null_dist)
    assert_allclose(test.statistic(x, y), Dcorr(bias=bias).statistic(x, y))


@pytest.mark.parametrize("initial_auto", [False, True])
@pytest.mark.parametrize("metric", ["euclidean", "cityblock", None, "precomputed"])
def test_statistic_after_test(initial_auto, metric):
    values = np.arange(24, dtype=float).reshape(-1, 1)
    x = np.hstack((values, np.cos(values)))
    y = np.hstack((np.sin(values), values % 5))
    if metric is None or metric == "precomputed":
        x, y = pairwise_distances(x), pairwise_distances(y)
    test = Dcorr(compute_distance=metric)
    test.test(x, y, auto=initial_auto, reps=9, random_state=1)

    expected = Dcorr(compute_distance=metric).statistic(x, y)
    assert_allclose(test.statistic(x, y), expected)
    assert_allclose(
        test.test(x, y, reps=9, random_state=1),
        Dcorr(compute_distance=metric).test(x, y, reps=9, random_state=1),
    )


@pytest.mark.parametrize("auto", [False, True])
def test_reuse_after_permutation_error(auto):
    x = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.sin(x)
    test = Dcorr()
    with pytest.raises(ValueError, match="n_jobs == 0"):
        test.test(x, y, auto=auto, reps=9, workers=0)

    assert not test.is_fast
    assert not test.is_distance
    assert_allclose(test.statistic(x, y), Dcorr().statistic(x, y))
