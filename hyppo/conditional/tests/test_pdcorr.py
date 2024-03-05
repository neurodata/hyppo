import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import indep_normal, power
from .. import PartialDcorr


class TestPDcorrStat:
    @pytest.mark.parametrize("n", [100, 200])
    @pytest.mark.parametrize("obs_stat", [0.0])
    def test_linear_oned(self, n, obs_stat):
        np.random.seed(123456789)
        x, y, z = indep_normal(n, 1)
        stat1, pvalue1 = PartialDcorr().test(x, y, z)
        stat2 = PartialDcorr().statistic(x, y, z)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(stat2, obs_stat, decimal=2)

    @pytest.mark.parametrize("n", [100, 200])
    def test_rep(self, n):
        x, y, z = indep_normal(n, 1)
        stat1, pvalue1 = PartialDcorr().test(x, y, z, random_state=2)
        stat2, pvalue2 = PartialDcorr().test(x, y, z, random_state=2)

        assert stat1 == stat2
        assert pvalue1 == pvalue2

    @pytest.mark.parametrize("n", [100])
    @pytest.mark.parametrize("obs_stat", [0.0143057])
    @pytest.mark.parametrize("obs_pvalue", [0.12])
    def test_indep_normal_corr(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y, z = indep_normal(n, 1)
        stat1, pvalue1 = PartialDcorr(use_cov=False).test(x, y, z)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(pvalue1, obs_pvalue, decimal=2)


class TestPDcorrTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "PartialDcorr",
            sim_type="condi",
            sim="independent_normal",
            n=100,
            p=1,
            alpha=0.05,
        )
        assert_almost_equal(est_power, 0.05, decimal=2)
