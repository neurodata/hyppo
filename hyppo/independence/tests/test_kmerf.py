import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_approx_equal, assert_raises

from ...tools import linear, multimodal_independence, power, spiral
from .. import KMERF


class TestKMERFStat(object):
    """Test validity of KMERF test statistic"""

    # commented out p-value calculation because build stalled
    @pytest.mark.parametrize(
        "sim, obs_stat, obs_pvalue",
        [
            (linear, 0.253, 1.0),  # test linear simulation
            (spiral, 0.037, 1.0),  # test spiral simulation
            (multimodal_independence, -0.0363, 1.0),  # test independence simulation
        ],
    )
    def test_oned(self, sim, obs_stat, obs_pvalue):
        np.random.seed(12345678)

        # generate x and y
        x, y = sim(n=100, p=1)

        # test stat and pvalue
        stat1 = KMERF().statistic(x, y)
        stat2, pvalue, _ = KMERF().test(x, y, reps=0)
        assert_approx_equal(stat1, obs_stat, significant=1)
        assert_approx_equal(stat2, obs_stat, significant=1)
        assert_approx_equal(pvalue, obs_pvalue, significant=1)


class TestKmerfErrorWarn:
    """Tests errors and warnings derived from MGC."""

    def test_no_indeptest(self):
        # raises error if not indep test
        assert_raises(ValueError, KMERF, "abcd")


# code too slow to run, check benchmarks in docs for verification
@pytest.mark.skip
class TestKmerfTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "KMERF",
            sim_type="indep",
            sim="multimodal_independence",
            n=5,
            p=1,
            alpha=0.05,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)
