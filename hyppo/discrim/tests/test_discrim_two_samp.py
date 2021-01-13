import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_warns

from .. import DiscrimTwoSample


@pytest.mark.skip(reason="reformat code to speed up test")
class TestTwoSample:
    def test_greater(self):
        # test whether discriminability for x1 is greater than it is for x2
        x1 = np.ones((100, 2), dtype=float)
        x2 = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        np.random.seed(123456789)
        obs_d1 = 0.5
        obs_d2 = 1.0
        obs_p = 1.0
        d1, d2, p = DiscrimTwoSample().test(x1, x2, y, alt="greater")

        assert_almost_equal(d1, obs_d1, decimal=2)
        assert_almost_equal(d2, obs_d2, decimal=2)
        assert_almost_equal(p, obs_p, decimal=2)

    def test_less(self):
        # test whether discriminability for x1 is less than it is for x2
        x1 = np.ones((100, 2), dtype=float)
        x2 = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        np.random.seed(123456789)
        obs_d1 = 0.5
        obs_d2 = 1.0
        obs_p = 0.000999000999000999
        d1, d2, p = DiscrimTwoSample().test(x1, x2, y, alt="less")

        assert_almost_equal(d1, obs_d1, decimal=2)
        assert_almost_equal(d2, obs_d2, decimal=2)
        assert_almost_equal(p, obs_p, decimal=4)

    def test_neq(self):
        # test whether discriminability for x1 is not equal compared to
        # discriminability for x2
        x1 = np.ones((100, 2), dtype=float)
        x2 = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        np.random.seed(123456789)
        obs_d1 = 0.5
        obs_d2 = 1.0
        obs_p = 0.000999000999000999
        d1, d2, p = DiscrimTwoSample().test(x1, x2, y, alt="neq")

        assert_almost_equal(d1, obs_d1, decimal=2)
        assert_almost_equal(d2, obs_d2, decimal=2)
        assert_almost_equal(p, obs_p, decimal=4)


class TestTwoSampleWarn:
    """
    Tests errors and warnings derived from one sample test.
    """

    def test_error_one_id(self):
        # checks whether y has only one id
        x1 = np.ones((100, 2), dtype=float)
        x2 = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.ones((100, 1))

        assert_raises(ValueError, DiscrimTwoSample().test, x1, x2, y)

    def test_error_unequal_row(self):
        # checks x1 & x2 for unequal number of instances
        x1 = np.ones((99, 2), dtype=float)
        x2 = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        assert_raises(ValueError, DiscrimTwoSample().test, x1, x2, y)

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        x1 = np.ones((100, 2), dtype=float)
        x2 = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        x1[0] = np.nan
        assert_raises(ValueError, DiscrimTwoSample().test, x1, x2, y)

        x1[0] = 1
        x2[0] = np.nan
        assert_raises(ValueError, DiscrimTwoSample().test, x1, x2, y)

        x2[0] = 1
        y[0] = np.nan
        assert_raises(ValueError, DiscrimTwoSample().test, x1, x2, y)

    def test_error_alt(self):
        # raises error if inputs contain NaNs
        x1 = np.ones((100, 2), dtype=float)
        x2 = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        x1[0] = np.nan
        assert_raises(ValueError, DiscrimTwoSample().test, x1, x2, y, alt="abc")

    @pytest.mark.parametrize(
        "reps", [-1, "1"]  # reps is negative  # reps is not integer
    )
    def test_error_reps(self, reps):
        # raises error if reps is negative
        x1 = np.ones((100, 2), dtype=float)
        x2 = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        assert_raises(ValueError, DiscrimTwoSample().test, x1, x2, y, reps=reps)

    def test_warns_reps(self):
        # raises warning when reps is less than 1000
        x1 = np.ones((100, 2), dtype=float)
        x2 = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        reps = 100
        assert_warns(RuntimeWarning, DiscrimTwoSample().test, x1, x2, y, reps=reps)
