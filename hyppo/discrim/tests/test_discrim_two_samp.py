import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_warns

from .. import DiscrimTwoSample


class TestTwoSample:
    def test_greater(self):
        # test whether discriminability for x1 is greater than it is for x2
        x1 = np.ones((100, 2), dtype=float)
        x2 = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        np.random.seed(123456789)
        obs_d1 = 0.5
        obs_d2 = 1.0
        d1, d2, _ = DiscrimTwoSample().test(x1, x2, y, alt="greater", reps=3)

        assert_almost_equal(d1, obs_d1, decimal=2)
        assert_almost_equal(d2, obs_d2, decimal=2)

    def test_less(self):
        # test whether discriminability for x1 is less than it is for x2
        x1 = np.ones((100, 2), dtype=float)
        x2 = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        np.random.seed(123456789)
        obs_d1 = 0.5
        obs_d2 = 1.0
        d1, d2, _ = DiscrimTwoSample().test(x1, x2, y, alt="less", reps=3)

        assert_almost_equal(d1, obs_d1, decimal=2)
        assert_almost_equal(d2, obs_d2, decimal=2)

    def test_neq(self):
        # test whether discriminability for x1 is not equal compared to
        # discriminability for x2
        x1 = np.ones((100, 2), dtype=float)
        x2 = np.concatenate((np.zeros((50, 2)), np.ones((50, 2))), axis=0)
        y = np.concatenate((np.zeros(50), np.ones(50)), axis=0)

        np.random.seed(123456789)
        obs_d1 = 0.5
        obs_d2 = 1.0
        d1, d2, _ = DiscrimTwoSample().test(x1, x2, y, alt="neq", reps=3)

        assert_almost_equal(d1, obs_d1, decimal=2)
        assert_almost_equal(d2, obs_d2, decimal=2)


class TestDiscrErrorWarn:
    """Tests errors and warnings."""

    def test_no_indeptest(self):
        # raises error if not indep test
        x1 = np.arange(20)
        x2 = np.arange(3, 23)
        y = np.arange(5, 25)
        assert_raises(ValueError, DiscrimTwoSample().test, x1, x2, y, alt="abcd")
