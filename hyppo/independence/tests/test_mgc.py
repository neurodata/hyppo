import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_approx_equal,
    assert_equal,
    assert_warns,
)
from scipy.stats import multiscale_graphcorr

from ...tools import linear, power, spiral
from .. import MGC


class TestMGCStat(object):
    """Test validity of MGC test statistic"""

    @pytest.mark.parametrize(
        "sim",
        (
            linear,  # test linear simulation
            spiral,  # test spiral simulation
        ),
    )
    def test_oned(self, sim):
        np.random.seed(12345678)

        # generate x and y
        x, y = sim(n=100, p=1)

        # test stat and pvalue
        stat_hyppo, pvalue_hyppo, _ = MGC().test(x, y)
        stat_scipy, pvalue_scipy, _ = multiscale_graphcorr(x, y)
        assert_approx_equal(stat_hyppo, stat_scipy, significant=4)
        assert_approx_equal(pvalue_hyppo, pvalue_scipy, significant=3)

    @pytest.mark.parametrize(
        "sim",
        [
            linear,  # test linear simulation
            spiral,  # test spiral simulation
        ],
    )
    def test_fived(self, sim):
        np.random.seed(12345678)

        # generate x and y
        x, y = sim(n=100, p=5)

        # test stat and pvalue
        stat_hyppo, pvalue_hyppo, _ = MGC().test(x, y)
        stat_scipy, pvalue_scipy, _ = multiscale_graphcorr(x, y)
        assert_approx_equal(stat_hyppo, stat_scipy, significant=4)
        assert_approx_equal(pvalue_hyppo, pvalue_scipy, significant=3)

    @pytest.mark.parametrize(
        "sim",
        [
            linear,  # test linear simulation
            spiral,  # test spiral simulation
        ],
    )
    def test_rep(self, sim):
        x, y = sim(n=100, p=5)
        stat1, pvalue1, _ = MGC().test(x, y, random_state=2)
        stat2, pvalue2, _ = MGC().test(x, y, random_state=2)

        assert stat1 == stat2
        assert pvalue1 == pvalue2


class TestMGCTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "MGC",
            sim_type="indep",
            sim="multimodal_independence",
            n=50,
            p=1,
            alpha=0.05,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)


class TestMGCErrorWarnings:
    def test_redundancy_warning(self):
        x = np.hstack((np.arange(0, 6), 5, 5, 5, 5, 5))
        y = np.hstack((np.arange(0, 6), 5, 5, 5, 5, 5))
        assert_warns(RuntimeWarning, MGC().test, x, y)
