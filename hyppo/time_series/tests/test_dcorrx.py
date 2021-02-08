import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...independence import Dcorr
from ...tools import cross_corr_ar, nonlinear_process
from .. import DcorrX


class TestDcorrXStat:
    def test_zero_var(self):
        x = np.ones(4)
        y = np.arange(4)
        stat = DcorrX().test(x, y)[0]
        assert_almost_equal(stat, 0.0)

    def test_multiple_lags(self):
        x = np.ones(6)
        y = np.arange(1, 7)
        stat = DcorrX(max_lag=3).test(x, y, reps=0)[0]
        assert_almost_equal(stat, 0.0)

    @pytest.mark.parametrize("n", [100, 200])
    @pytest.mark.parametrize("obs_pvalue", [1 / 1000])
    @pytest.mark.parametrize("obs_opt_lag", [1])
    def test_correlated_oned(self, n, obs_pvalue, obs_opt_lag):
        np.random.seed(123456789)

        x, y = cross_corr_ar(n, lag=1, phi=0.9, sigma=0.1)
        _, pvalue, dcorrx_dict = DcorrX(max_lag=1).test(x, y)
        opt_lag = dcorrx_dict["opt_lag"]

        assert_almost_equal(pvalue, obs_pvalue, decimal=2)
        assert_almost_equal(opt_lag, obs_opt_lag, decimal=0)

    def test_lag0(self):
        # Dependent
        x, y = nonlinear_process(100, lag=1)

        stat1 = Dcorr().test(x, y, auto=False, reps=0)[0]
        stat2 = DcorrX(max_lag=0).test(x, y, reps=0)[0]

        assert_almost_equal(stat1, stat2, decimal=2)

        # Independent
        x = np.array([[0.95476444], [0.36741182], [0.37473209], [-0.62469378]])
        y = np.array([[0.86515308], [-0.01985114], [0.21544295], [1.56443374]])
        stat = Dcorr().test(x, y, auto=False, reps=0)[0]
        statx = DcorrX(max_lag=0).test(x, y, reps=0)[0]

        assert_almost_equal(stat, statx, decimal=2)
