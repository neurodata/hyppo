import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_warns

from ...tools import linear, power
from .. import FriedmanRafsky


class TestFriedmanRafskyStat:
    @pytest.mark.parametrize("n", [200, 300])
    @pytest.mark.parametrize("obs_stat", [-0.50])
    @pytest.mark.parametrize("obs_pvalue", [1 / 1000])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        num_rows, num_cols = x.shape
        y = np.random.choice([0, 1], num_rows, p=[0.5, 0.5])
        stat1, pvalue1 = FriedmanRafsky().test(x, y)
        stat2 = FriedmanRafsky().statistic(x, y)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(stat2, obs_stat, decimal=2)
        assert_almost_equal(pvalue1, obs_pvalue, decimal=2)

    @pytest.mark.parametrize("n", [100, 200])
    def test_rep(self, n):
        x, y = linear(n, 1)
        num_rows, num_cols = x.shape
        y = np.random.choice([0, 1], num_rows, p=[0.5, 0.5])
        stat1, pvalue1 = FriedmanRafsky().test(x, y, random_state=2)
        stat2, pvalue2 = FriedmanRafsky().test(x, y, random_state=2)

        assert stat1 == stat2
        assert pvalue1 == pvalue2
