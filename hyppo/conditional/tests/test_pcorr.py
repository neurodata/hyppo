import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import indep_normal, power
from .. import PartialCorr


class TestPcorrStat:
    @pytest.mark.parametrize("n", [100, 200])
    @pytest.mark.parametrize("obs_stat", [0.0])
    def test_linear_oned(self, n, obs_stat):
        np.random.seed(3)
        x, y, z = indep_normal(n, 1)
        stat1, pvalue1 = PartialCorr().test(x, y, z)
        stat2 = PartialCorr().statistic(x, y, z)

        assert_almost_equal(stat1, obs_stat, decimal=1)
        assert_almost_equal(stat2, obs_stat, decimal=1)

    @pytest.mark.parametrize("n", [100, 200])
    def test_rep(self, n):
        x, y, z = indep_normal(n, 1)
        stat1, pvalue1 = PartialCorr().test(
            x, y, z, random_state=2, auto=False, reps=1000
        )
        stat2, pvalue2 = PartialCorr().test(
            x, y, z, random_state=2, auto=False, reps=1000
        )

        assert stat1 == stat2
        assert pvalue1 == pvalue2


class TestPcorrTypeIError:
    def test_oned(self):
        np.random.seed(1)
        est_power = power(
            "PartialCorr",
            sim_type="condi",
            sim="independent_normal",
            n=100,
            p=1,
            alpha=0.05,
        )
        assert_almost_equal(0.05, est_power, decimal=2)
