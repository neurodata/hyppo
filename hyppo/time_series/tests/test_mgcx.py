import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from ...independence import MGC
from .. import MGCX
from ...sims import cross_corr_ar, nonlinear_process


class TestMGCXStat:
    def test_zero_var(self):
        n = 10
        x = np.ones(n).reshape(n, 1)
        y = np.arange(n).reshape(n, 1)
        stat = MGCX().test(x, y)[0]
        assert_almost_equal(stat, 0.0)

    def test_multiple_lags(self):
        n = 10
        x = np.ones(n).reshape(n, 1)
        y = np.arange(1, n + 1).reshape(n, 1)
        stat = MGCX(max_lag=3).test(x, y)[0]
        assert_almost_equal(stat, 0.0)

    # def test_nonlinear_oned(self, n, obs_pvalue, obs_opt_lag):
    #     np.random.seed(123456789)

    #     x, y = nonlinear_process(120, lag=1)
    #     _, pvalue, mgcx_dict = MGCX(max_lag=1).test(x, y)
    #     opt_lag = mgcx_dict["opt_lag"]

    #     assert_almost_equal(pvalue, 1 / 1000, decimal=2)
    #     assert_almost_equal(opt_lag, 1, decimal=0)

    # def test_lag0(self):
    #     x, y = cross_corr_ar(20, lag=1, phi=0.9)

    #     stat1 = MGC().test(x, y)[0]
    #     stat2 = MGCX(max_lag=0).test(x, y)[0]

    #     assert_almost_equal(stat1, stat2, decimal=0)

    # def test_distance(self):
    #     n = 10
    #     x = np.ones(n)
    #     y = np.arange(1, n + 1)

    #     distx = np.zeros((n, n))
    #     disty = np.fromfunction(lambda i, j: np.abs(i - j), (n, n))

    #     stat1 = MGCX(max_lag=1).test(x, y)[0]
    #     stat2 = MGCX(max_lag=1).test(distx, disty)[0]

    #     assert_almost_equal(stat1, stat2, decimal=0)

