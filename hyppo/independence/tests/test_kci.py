import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_warns

from ...tools import linear, power
from .. import KCI


class TestKCI:
    @pytest.mark.parametrize("obs_stat", [0])
    @pytest.mark.parametrize("obs_pvalue", [1])
    def test_linear_oned(self, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x = np.random.choice([0, 1], (100, 2), p=[0.5, 0.5])
        y = np.random.choice([0, 1], (100, 2), p=[0.5, 0.5])
        stat1, pvalue1 = KCI().statistic(x, y)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(pvalue1, obs_pvalue, decimal=2)

    @pytest.mark.parametrize("n", [100, 200])
    def test_rep(self, n):
        x = np.random.choice([0, 1], (100, 2), p=[0.5, 0.5])
        y = np.random.choice([0, 1], (100, 2), p=[0.5, 0.5])
        stat1, pvalue1 = KCI().statistic(x, y, random_state=2)
        stat2, pvalue2 = KCI().statistic(x, y, random_state=2)

        assert stat1 == stat2
        assert pvalue1 == pvalue2
