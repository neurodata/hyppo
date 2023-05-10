import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import cross_corr_ar, nonlinear_process
from .. import LjungBox


class TestLjungBox:
    def test_invalid_inputs(self):
        x = np.ones(5)
        y = np.ones(5)
        with pytest.raises(ValueError):
            LjungBox(max_lag=0).test(x, y)

    def test_statistic_test_functions(self):
        np.random.seed(1)

        n = 20
        x = np.arange(1, n + 1).reshape(n, 1)
        y = np.arange(1, n + 1).reshape(n, 1)

        stat = LjungBox(max_lag=1).statistic(x, y)[0]
        stat2 = LjungBox(max_lag=1).test(x, y)[0]

        assert stat == stat2

    def test_same_seq(self):
        np.random.seed(1)

        n = 20
        x = np.arange(1, n + 1).reshape(n, 1)
        y = np.arange(1, n + 1).reshape(n, 1)
        stat = LjungBox(max_lag=1).statistic(x, y)[0]
        assert_almost_equal(stat, 18.53, decimal=2)

        res = LjungBox(max_lag=1).test(x, y)
        assert_almost_equal(res[1], 0.0, decimal=2)

    def test_same_seq_permutations(self):
        np.random.seed(2)

        n = 20
        x = np.arange(1, n + 1).reshape(n, 1)
        y = np.arange(1, n + 1).reshape(n, 1)
        res = LjungBox(max_lag=1).test(x, y, auto=False)
        assert_almost_equal(res[1], 0.0, decimal=2)

    def test_linear(self):
        np.random.seed(3)

        x, y = cross_corr_ar(100, lag=1)
        _, pvalue, mgcx_dict = LjungBox(max_lag=5).test(x, y)
        opt_lag = mgcx_dict["opt_lag"]

        assert_almost_equal(pvalue, 1 / 1000, decimal=2)
        assert_almost_equal(opt_lag, 1, decimal=0)

    def test_nonlinear(self):
        np.random.seed(4)

        x, y = nonlinear_process(100, lag=1)
        _, pvalue, mgcx_dict = LjungBox(max_lag=1).test(x, y, auto=False)
        opt_lag = mgcx_dict["opt_lag"]

        assert_almost_equal(pvalue, 0.9, decimal=1)
        assert_almost_equal(opt_lag, 1, decimal=0)
