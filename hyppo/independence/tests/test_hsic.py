import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.metrics.pairwise import pairwise_kernels

from ...tools import linear, power
from .. import Hsic


class TestHsicStat:
    @pytest.mark.parametrize("n, obs_stat", [(100, 1.0), (200, 1.0)])
    @pytest.mark.parametrize("obs_pvalue", [1 / 1000])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        stat, pvalue = Hsic().test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=2)

    def test_kernstat(self):
        np.random.seed(123456789)
        x, y = linear(100, 1)
        kernx = pairwise_kernels(x, x)
        kerny = pairwise_kernels(y, y)
        stat, pvalue = Hsic(compute_kernel=None).test(kernx, kerny, auto=False)

        assert_almost_equal(stat, 1.0, decimal=2)
        assert_almost_equal(pvalue, 1 / 1000, decimal=2)


class TestHsicTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "Hsic",
            sim_type="indep",
            sim="multimodal_independence",
            n=200,
            p=1,
            alpha=0.05,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)

    def test_oned_fast(self):
        np.random.seed(123456789)
        est_power = power(
            "Hsic",
            sim_type="indep",
            sim="multimodal_independence",
            n=500,
            p=1,
            alpha=0.05,
            auto=True,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)
