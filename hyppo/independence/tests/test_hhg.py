import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.metrics import pairwise_distances

from ...tools import linear, power
from .. import HHG


class TestHHGStat:
    @pytest.mark.parametrize(
        "n, obs_stat", [(10, 560.0), (50, 112800.0), (100, 950600.0)]
    )
    @pytest.mark.parametrize("obs_pvalue", [1 / 1000])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        stat, pvalue = HHG().test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=2)

    def test_diststat(self):
        np.random.seed(123456789)
        x, y = linear(100, 1)
        distx = pairwise_distances(x, x)
        disty = pairwise_distances(y, y)
        stat = HHG().statistic(distx, disty)

        assert_almost_equal(stat, 950600.0, decimal=2)


class TestHHGTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "HHG",
            sim_type="indep",
            sim="multimodal_independence",
            n=50,
            p=1,
            alpha=0.05,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)
