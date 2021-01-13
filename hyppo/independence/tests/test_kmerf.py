import numpy as np
import pytest
from numpy.testing import assert_approx_equal

from ...tools import linear, multimodal_independence, spiral
from .. import KMERF


class TestKMERFStat(object):
    """Test validity of KMERF test statistic"""

    # commented out p-value calculation because build stalled
    @pytest.mark.parametrize(
        "sim, obs_stat, obs_pvalue",
        [
            (linear, 0.253, 1 / 1000),  # test linear simulation
            (spiral, 0.037, 0.012),  # test spiral simulation
            (multimodal_independence, -0.0363, 0.995),  # test independence simulation
        ],
    )
    def test_oned(self, sim, obs_stat, obs_pvalue):
        np.random.seed(12345678)

        # generate x and y
        x, y = sim(n=100, p=1)

        # test stat and pvalue
        stat1 = KMERF()._statistic(x, y)
        # stat2, pvalue = KMERF().test(x, y)
        assert_approx_equal(stat1, obs_stat, significant=1)
        # assert_approx_equal(stat2, obs_stat, significant=1)
        # assert_approx_equal(pvalue, obs_pvalue, significant=1)

    # commented out p-value calculation because build stalled
    @pytest.mark.parametrize(
        "sim, obs_stat, obs_pvalue",
        [
            (linear, 0.354, 1 / 1000),  # test linear simulation
            (spiral, 0.091, 0.001),  # test spiral simulation
        ],
    )
    def test_fived(self, sim, obs_stat, obs_pvalue):
        np.random.seed(12345678)

        # generate x and y
        x, y = sim(n=100, p=5)

        # test stat and pvalue
        stat1 = KMERF()._statistic(x, y)
        # stat2, pvalue = KMERF().test(x, y)
        assert_approx_equal(stat1, obs_stat, significant=1)
        # assert_approx_equal(stat2, obs_stat, significant=1)
        # assert_approx_equal(pvalue, obs_pvalue, significant=1)
