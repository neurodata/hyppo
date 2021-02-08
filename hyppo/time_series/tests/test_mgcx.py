import numpy as np
from numpy.testing import assert_almost_equal, assert_array_less

from ...tools import nonlinear_process
from .. import MGCX


class TestMGCXStat:
    def test_multiple_lags(self):
        n = 10
        x = np.arange(1, n + 1).reshape(n, 1)
        y = np.arange(1, n + 1).reshape(n, 1)
        pvalue = MGCX(max_lag=3).test(x, y)[1]
        assert_almost_equal(pvalue, 1 / 1000, decimal=2)

    def test_nonlinear(self):
        np.random.seed(123456789)

        x, y = nonlinear_process(120, lag=1)
        _, pvalue, mgcx_dict = MGCX(max_lag=1).test(x, y)
        opt_lag = mgcx_dict["opt_lag"]

        assert_almost_equal(pvalue, 1 / 1000, decimal=2)
        assert_almost_equal(opt_lag, 1, decimal=0)

    # TO DO: Fix when repeated values are fixed in classic MGC.
    # def test_lag0(self):
    #     x, y = cross_corr_ar(20, lag=1, phi=0.9)

    #     stat1 = MGC().test(x, y)[0]
    #     stat2 = MGCX(max_lag=0).test(x, y)[0]

    #     assert_almost_equal(stat1, stat2, decimal=0)

    def test_optimal_scale_linear(self):
        n = 10
        x = np.arange(n)
        y = x
        mgcx_dict = MGCX().test(x, y, reps=100)[2]
        opt_scale = mgcx_dict["opt_scale"]
        assert_almost_equal(opt_scale[0], n, decimal=0)
        assert_almost_equal(opt_scale[1], n, decimal=0)

    def test_optimal_scale_nonlinear(self):
        n = 7
        x = np.arange(n)
        y = x ** 2
        mgcx_dict = MGCX().test(x, y, reps=100)[2]
        opt_scale = mgcx_dict["opt_scale"]
        assert_array_less(opt_scale[0], n)
        assert_array_less(opt_scale[1], n)
