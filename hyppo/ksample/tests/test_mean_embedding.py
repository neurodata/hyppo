import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from hyppo.tools import rot_ksamp
from hyppo.ksample import MeanEmbeddingTest


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
        stat, pvalue = MeanEmbeddingTest().test(x, y, random_state=1234)
        print(stat, pvalue)
        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=10)

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
        stat, pvalue = MeanEmbeddingTest(num_randfreq=1).test(x, y, random_state=1234)
        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pval, decimal=2)
