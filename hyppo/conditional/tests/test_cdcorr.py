import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises

from ...tools import indep_normal, power
from .. import ConditionalDcorr


class TestCDcorrStatPvalue:
    @pytest.mark.parametrize("n", [50])
    @pytest.mark.parametrize("obs_stat", [0.0])
    @pytest.mark.parametrize("bandwidth", [None, "silverman", np.ones(1)])
    def test_indep_normal(self, n, obs_stat, bandwidth):
        np.random.seed(123456789)
        x, y, z = indep_normal(n, 1)
        stat1, pvalue1 = ConditionalDcorr(bandwidth=bandwidth).test(x, y, z)
        stat2 = ConditionalDcorr(bandwidth=bandwidth).statistic(x, y, z)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(stat2, obs_stat, decimal=2)

    @pytest.mark.parametrize("n", [50, 100])
    def test_rep(self, n):
        x, y, z = indep_normal(n, 1)
        stat1, pvalue1 = ConditionalDcorr().test(x, y, z, random_state=2)
        stat2, pvalue2 = ConditionalDcorr().test(x, y, z, random_state=2)

        assert stat1 == stat2
        assert pvalue1 == pvalue2

    @pytest.mark.parametrize("n", [100])
    @pytest.mark.parametrize("obs_stat", [0.12926180100498708])
    @pytest.mark.parametrize("obs_pvalue", [0.06393606393606394])
    def test_indep_normal_corr(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y, z = indep_normal(n, 1)
        stat1, pvalue1 = ConditionalDcorr(use_cov=False).test(x, y, z)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(pvalue1, obs_pvalue, decimal=2)


class TestCDcorrErrors:
    def test_bandwith(self):
        with pytest.raises(ValueError):
            ConditionalDcorr(bandwidth="bad_bandwidth")


class TestCDcorrTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "ConditionalDcorr",
            sim_type="condi",
            sim="independent_normal",
            n=100,
            p=1,
            alpha=0.05,
        )
        assert_almost_equal(0.05, est_power, decimal=2)
