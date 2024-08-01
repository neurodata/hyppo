import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_warns

from ...tools import linear, logarithmic, power
from .. import KCI


class TestKCI:
    @pytest.mark.parametrize("n", [100])
    @pytest.mark.parametrize("obs_stat", [544.691])
    @pytest.mark.parametrize("obs_pvalue", [0])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        stat1, pvalue1 = KCI().test(x, y)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(pvalue1, obs_pvalue, decimal=2)

    @pytest.mark.parametrize("n", [100])
    @pytest.mark.parametrize("obs_stat", [88.526])
    @pytest.mark.parametrize("obs_pvalue", [.00027])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = logarithmic(n, 1)
        stat1, pvalue1 = KCI().test(x, y)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(pvalue1, obs_pvalue, decimal=2)

    @pytest.mark.parametrize("n", [100, 200])
    def test_rep(self, n):
        x, y = linear(n, 1)
        stat1, pvalue1 = KCI().test(x, y)
        stat2, pvalue2 = KCI().test(x, y)

        assert stat1 == stat2
        assert pvalue1 == pvalue2
