import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises
from sklearn.metrics import pairwise_distances

from ...tools import power, rot_ksamp
from .. import DISCO


class TestDISCO:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue",
        [(200, 6.621905272534802, 0.001), (100, 2.675357570989666, 0.001)],
    )
    def test_disco_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2)
        stat, pvalue = DISCO().test(x, y, auto=False)

        assert_almost_equal(stat, obs_stat, decimal=1)
        assert_almost_equal(pvalue, obs_pvalue, decimal=1)


class TestDISCOErrorWarn:
    """Tests errors and warnings derived from MGC."""

    def test_diffshape(self):
        # raises error if not indep test
        x = np.arange(20)
        y = np.arange(10)
        assert_raises(ValueError, DISCO().statistic, x, y)
        assert_raises(ValueError, DISCO().test, x, y)


class TestDISCOTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "DISCO",
            sim_type="ksamp",
            sim="multimodal_independence",
            k=2,
            n=100,
            p=1,
            alpha=0.05,
            auto=True,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)
