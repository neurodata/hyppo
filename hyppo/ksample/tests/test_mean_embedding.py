import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from hyppo.tools import rot_ksamp
from hyppo.ksample import MeanEmbeddingTest
import hyppo.ksample.mean_embedding


class TestMeanEmbedding:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue",
        [
            (200, 77.452, 2.862e-15),
            (700, 245.478, 5.132e-51),
        ],
    )
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2)
        stat, pvalue = MeanEmbeddingTest(random_state=1234).test(x, y)
        print(stat, pvalue)
        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=10)

    @pytest.mark.parametrize(
        "actual_norm",
        [
            ([0.00637311, 0.59369549, 0.16233412, 0.32396812, 0.05284812]),
        ],
    )
    def test_get_estimate(self, actual_norm):
        np.random.seed(123456789)
        x = np.random.randn(5, 2)
        point = np.random.randn(5, 2)
        norm = hyppo.ksample.mean_embedding._get_estimate(x, point)
        assert_almost_equal(norm, actual_norm)

    @pytest.mark.parametrize(
        "actual_diff",
        [
            ([-0.2081915, -0.37618035, 0.70235683, -0.7459787, 0.44686362]),
        ],
    )
    def test_get_difference(self, actual_diff):
        np.random.seed(123456789)
        x = np.random.randn(5, 2)
        y = np.random.randn(5, 2)
        point = np.random.randn(5, 2)
        diff = hyppo.ksample.mean_embedding._get_difference(point, x, y)
        assert_almost_equal(diff, actual_diff)

    def test_null(self):
        np.random.seed(120)
        x = np.random.randn(500, 10)
        y = np.random.randn(500, 10)
        _, pval = MeanEmbeddingTest().test(x, y)
        assert pval > 0.05

    def test_alternative(self):
        np.random.seed(120)
        x = np.random.randn(500, 10)
        x[:, 1] *= 3
        y = np.random.randn(500, 10)
        _, pval = MeanEmbeddingTest().test(x, y)
        assert pval < 0.05

    @pytest.mark.parametrize(
        "obs_stat, obs_pval",
        [
            (0.0164, 0.8980),
        ],
    )
    def test_one_frequency(self, obs_stat, obs_pval):
        np.random.seed(123456789)
        x = np.random.randn(500, 10)
        y = np.random.randn(500, 10)
        stat, pvalue = MeanEmbeddingTest(random_state=1234, num_randfreq=1).test(x, y)
        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pval, decimal=2)
