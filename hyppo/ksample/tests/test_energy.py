import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises

from ...tools import linear, rot_ksamp
from .. import Energy


class TestEnergy:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue",
        [(200, 4.28e-7, 0.001), (100, 8.24e-5, 0.001)],
    )
    def test_energy_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = rot_ksamp(linear, n, 1, k=2)
        stat, pvalue = Energy().test(x, y, auto=False)

        assert_almost_equal(stat, obs_stat, decimal=1)
        assert_almost_equal(pvalue, obs_pvalue, decimal=1)


class TestEnergyErrorWarn:
    """Tests errors and warnings derived from MGC."""

    def test_error_notndarray(self):
        # raises error if x or y is not a ndarray
        x = np.arange(20)
        y = [5] * 20
        z = np.arange(5)
        assert_raises(ValueError, Energy().test, x, y, z)

    def test_error_shape(self):
        # raises error if number of samples different (n)
        x = np.arange(100).reshape(25, 4)
        y = x.reshape(10, 10)
        z = x
        assert_raises(ValueError, Energy().test, x, y, z)

    def test_error_lowsamples(self):
        # raises error if samples are low (< 3)
        x = np.arange(3)
        y = np.arange(3)
        assert_raises(ValueError, Energy().test, x, y)

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        x = np.arange(20, dtype=float)
        x[0] = np.nan
        assert_raises(ValueError, Energy().test, x, x)

        y = np.arange(20)
        assert_raises(ValueError, Energy().test, x, y)
