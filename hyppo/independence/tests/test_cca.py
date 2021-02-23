import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import joint_normal, linear, power
from .. import CCA


class TestCCAStat:
    @pytest.mark.parametrize("n", [10, 100, 1000])
    @pytest.mark.parametrize("obs_stat", [1.0])
    @pytest.mark.parametrize("obs_pvalue", [1 / 1000])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        stat, pvalue = CCA().test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=2)

    @pytest.mark.parametrize("n", [100, 1000, 10000])
    @pytest.mark.parametrize("obs_stat", [0.07])
    @pytest.mark.parametrize("obs_pvalue", [1 / 1000])
    def test_linear_threed(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = joint_normal(n, 3)
        stat, pvalue = CCA().test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=1)
        assert_almost_equal(pvalue, obs_pvalue, decimal=1)


class TestCCATypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "CCA",
            sim_type="indep",
            sim="multimodal_independence",
            n=1000,
            p=1,
            alpha=0.05,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)
