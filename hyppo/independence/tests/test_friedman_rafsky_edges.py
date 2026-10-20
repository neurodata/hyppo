import numpy as np
import pytest
from numpy.testing import assert_array_equal

from .. import FriedmanRafsky


@pytest.mark.parametrize("values", [[0, 1, 2, 3], [0, 0, 1, 2]])
@pytest.mark.parametrize("scale", [1.0, 1e8])
def test_minimum_tree_runs(values, scale):
    # The two classes occupy consecutive intervals on the line. Every minimum
    # tree has exactly one edge between classes, including with repeated points.
    x = scale * np.asarray(values, dtype=float)[:, None]
    labels = np.array([0, 0, 1, 1])

    assert FriedmanRafsky().statistic(x, labels) == 2


def test_coincident_points_remain_connected():
    x = np.zeros((4, 2))
    labels = np.array([0, 0, 1, 1])

    # Ties allow multiple minimum trees, but separating both classes must
    # produce at least two runs, and there cannot be more runs than points.
    assert 2 <= FriedmanRafsky().statistic(x, labels) <= 4


def test_large_distances_preserve_permutation_result():
    x = np.arange(12, dtype=float)[:, None]
    labels = np.tile([0, 0, 1, 1], 3)
    expected = FriedmanRafsky().test(x, labels, reps=99, random_state=0)
    actual = FriedmanRafsky().test(1e8 * x, labels, reps=99, random_state=0)

    assert np.isfinite(expected).all()
    assert_array_equal(actual, expected)
