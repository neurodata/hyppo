import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.metrics import pairwise_distances

from ...tools import linear, power
from .. import HHG
from ..hhg import hoeffdings


class TestFastHHGStat:
    @pytest.mark.parametrize("n, obs_stat", [(10, 1.0), (50, 1.0), (100, 1.0)])
    @pytest.mark.parametrize("obs_pvalue", [1 / 1000])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        stat, pvalue = HHG().test(x, y, auto=True)

        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=2)

    def test_diststat(self):
        np.random.seed(123456789)
        x, y = linear(100, 1)
        zx = np.mean(x, axis=0).reshape(1, -1)
        zy = np.mean(y, axis=0).reshape(1, -1)
        distx = pairwise_distances(zx, x).reshape(-1, 1)
        disty = pairwise_distances(zy, y).reshape(-1, 1)
        test = HHG(compute_distance=None)
        test.auto = True
        stat = test.statistic(distx, disty)

        assert_almost_equal(stat, 1.0, decimal=2)

    def test_stat(self):
        np.random.seed(123456789)
        x, y = linear(100, 1)
        test = HHG()
        test.auto = True
        stat = test.statistic(x, y)

        assert_almost_equal(stat, 1.0, decimal=2)

    @pytest.mark.parametrize("n", [10, 200])
    def test_rep(self, n):
        x, y = linear(n, 1)
        stat, pvalue = HHG().test(x, y, auto=True, random_state=2)
        stat2, pvalue2 = HHG().test(x, y, auto=True, random_state=2)

        assert stat == stat2
        assert pvalue == pvalue2
