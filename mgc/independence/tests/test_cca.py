import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from ...benchmarks.simulation import linear
from ..cca import CannCorr


class TestCCAStat:
    @pytest.mark.parametrize("n", [10, 100, 1000])
    @pytest.mark.parametrize("obs_stat", [1.0])
    @pytest.mark.parametrize("obs_pvalue", [1/1000])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed()
        x, y = linear(n, 1, dim=1, noise=0)
        reps = 1000
        cca = CannCorr()
        stat = cca.statistic(x, y)
        pvalue = cca.p_value(x, y, reps=reps)[0]

        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=2)

    @pytest.mark.parametrize("n", [10, 100, 1000])
    @pytest.mark.parametrize("obs_stat", [1.0])
    @pytest.mark.parametrize("obs_pvalue", [1/1000])
    def test_linear_fived(self, n, obs_stat, obs_pvalue):
        np.random.seed()
        x, y = linear(n, 1, dim=5, noise=0)
        reps = 1000
        cca = CannCorr()
        stat = cca.statistic(x, y)
        pvalue = cca.p_value(x, y, reps=reps)[0]

        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=2)
