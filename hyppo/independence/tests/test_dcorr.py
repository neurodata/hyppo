import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_warns

from ...tools import linear
from .. import Dcorr


class TestDcorrStat:
    @pytest.mark.parametrize("n", [100, 200])
    @pytest.mark.parametrize("obs_stat", [1.0])
    @pytest.mark.parametrize("obs_pvalue", [1 / 1000])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        stat1, pvalue1 = Dcorr().test(x, y)
        stat2 = Dcorr().statistic(x, y)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(stat2, obs_stat, decimal=2)
        assert_almost_equal(pvalue1, obs_pvalue, decimal=2)


class TestDcorrErrorWarn:
    """Tests errors and warnings derived from MGC."""

    def test_error_notndarray(self):
        # raises error if x or y is not a ndarray
        x = np.arange(20)
        y = [5] * 20
        assert_raises(TypeError, Dcorr().test, x, y)
        assert_raises(TypeError, Dcorr().test, y, x)

    def test_error_shape(self):
        # raises error if number of samples different (n)
        x = np.arange(100).reshape(25, 4)
        y = x.reshape(10, 10)
        assert_raises(ValueError, Dcorr().test, x, y)

    def test_error_lowsamples(self):
        # raises error if samples are low (< 3)
        x = np.arange(3)
        y = np.arange(3)
        assert_raises(ValueError, Dcorr().test, x, y)

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        x = np.arange(20, dtype=float)
        x[0] = np.nan
        assert_raises(ValueError, Dcorr().test, x, x)

        y = np.arange(20)
        assert_raises(ValueError, Dcorr().test, x, y)

    def test_error_wrongdisttype(self):
        # raises error if compute_distance is not a function
        x = np.arange(20)
        compute_distance = 1
        dcorr = Dcorr(compute_distance=compute_distance)
        assert_raises(ValueError, dcorr.test, x, x)

    @pytest.mark.parametrize(
        "reps", [-1, "1"]  # reps is negative  # reps is not integer
    )
    def test_error_reps(self, reps):
        # raises error if reps is negative
        x = np.arange(20)
        assert_raises(ValueError, Dcorr().test, x, x, reps=reps)
