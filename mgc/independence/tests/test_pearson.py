import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_raises, assert_warns

from ...sims import linear
from .. import Pearson


class TestPearsonStat:
    @pytest.mark.parametrize("n", [10, 100, 1000])
    @pytest.mark.parametrize("obs_stat", [1.0])
    @pytest.mark.parametrize("obs_pvalue", [1/1000])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        stat, pvalue = Pearson().test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=2)


class TestPearsonErrorWarn:
    """ Tests errors and warnings derived from MGC.
    """
    def test_error_notndarray(self):
        # raises error if x or y is not a ndarray
        x = np.arange(20)
        y = [5] * 20
        assert_raises(ValueError, Pearson().test, x, y)
        assert_raises(ValueError, Pearson().test, y, x)

    def test_error_lowsamples(self):
        # raises error if samples are low (< 3)
        x = np.arange(3)
        y = np.arange(3)
        assert_raises(ValueError, Pearson().test, x, y)

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        x = np.arange(20, dtype=float)
        x[0] = np.nan
        assert_raises(ValueError, Pearson().test, x, x)

        y = np.arange(20)
        assert_raises(ValueError, Pearson().test, x, y)
