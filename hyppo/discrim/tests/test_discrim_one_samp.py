import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_warns

from .. import DiscrimOneSample


class TestOneSample:
    def test_same_one(self):
        # matches test calculated statistics and p-value for indiscriminable subjects
        x = np.ones((100, 2), dtype=float)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        np.random.seed(123456789)
        obs_stat = 0.5
        obs_p = 1
        stat, p = DiscrimOneSample().test(x, y, reps=0)

        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(p, obs_p, decimal=2)

    def test_diff_one(self):
        # matches test calculated statistics and p-value for discriminable subjects
        x = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        np.random.seed(123456789)
        obs_stat = 1.0
        obs_p = 1.0
        stat, p = DiscrimOneSample().test(x, y, reps=0)

        assert_almost_equal(stat, obs_stat, decimal=3)
        assert_almost_equal(p, obs_p, decimal=3)


class TestOneSampleWarn:
    """Tests errors and warnings derived from one sample test."""

    def test_error_one_id(self):
        # checks whether y has only one id
        X = np.ones((100, 2), dtype=float)
        Y = np.ones((100, 1))

        assert_raises(ValueError, DiscrimOneSample().test, X, Y)

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        x = np.arange(20, dtype=float)
        x[0] = np.nan
        assert_raises(ValueError, DiscrimOneSample().test, x, x)

        y = np.arange(20)
        assert_raises(ValueError, DiscrimOneSample().test, x, y)

    @pytest.mark.parametrize(
        "reps", [-1, "1"]  # reps is negative  # reps is not integer
    )
    def test_error_reps(self, reps):
        # raises error if reps is negative
        x = np.ones((100, 2), dtype=float)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        assert_raises(ValueError, DiscrimOneSample().test, x, y, reps=reps)

    def test_min_sample(self):
        # raises warning when sample number is less than 10
        x = np.ones((6, 2), dtype=float)
        y = np.concatenate((np.zeros(3), np.ones(3)), axis=0)

        assert_raises(ValueError, DiscrimOneSample().test, x, y)
