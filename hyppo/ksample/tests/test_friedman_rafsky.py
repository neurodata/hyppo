import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_warns

from ...tools import linear, power
from .. import FriedmanRafsky


class TestFriedmanRafskyStat:
    @pytest.mark.parametrize("n", [100])
    @pytest.mark.parametrize("num_runs", [48])
    @pytest.mark.parametrize("obs_stat", [-0.527])
    @pytest.mark.parametrize("obs_pvalue", [0.748])
    def test_linear_oned(self, n, num_runs, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        num_rows, num_cols = x.shape
        y = np.random.choice([0, 1], num_rows, p=[0.5, 0.5])
        y = np.transpose(y)
        stat1, pvalue1, _ = FriedmanRafsky().test(x, y)
        stat2 = FriedmanRafsky().statistic(x, y)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(stat2, num_runs, decimal=2)
        assert_almost_equal(pvalue1, obs_pvalue, decimal=2)

    @pytest.mark.parametrize("n", [100, 200])
    def test_rep(self, n):
        x, y = linear(n, 1)
        num_rows, num_cols = x.shape
        y = np.random.choice([0, 1], num_rows, p=[0.5, 0.5])
        y = np.transpose(y)
        stat1, pvalue1, _ = FriedmanRafsky().test(x, y, random_state=2)
        stat2, pvalue2, _ = FriedmanRafsky().test(x, y, random_state=2)

        assert stat1 == stat2
        assert pvalue1 == pvalue2
