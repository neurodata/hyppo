import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_warns

from ...tools import linear, power
from .. import RV


class TestRVStat:
    @pytest.mark.parametrize("n", [10, 100, 1000])
    @pytest.mark.parametrize("obs_stat", [1.0])
    @pytest.mark.parametrize("obs_pvalue", [1 / 1000])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        stat, pvalue = RV().test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=2)


class TestCCATypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "RV",
            sim_type="indep",
            sim="multimodal_independence",
            n=1000,
            p=1,
            alpha=0.05,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)
