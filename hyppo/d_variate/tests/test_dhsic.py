import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import linear, power
from .. import dHsic  # type: ignore


class TestdHsicStat:
    @pytest.mark.parametrize("n, obs_stat", [(100, 0.04561), (200, 0.03911)])
    @pytest.mark.parametrize("obs_pvalue", [1 / 1000])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        stat, pvalue = dHsic(gamma=0.5).test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=2)


class TestdHsicTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "dhsic",
            sim_type="multi",
            sim="multimodal_independence",
            n=100,
            p=1,
            alpha=0.05,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)
