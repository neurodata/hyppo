import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from hyppo.tools import rot_ksamp
from hyppo.ksample import MeanEmbeddingTest
from hyppo.ksample.mean_embedding import _get_estimate


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
    def test_norm(self, actual_norm):
        np.random.seed(123456789)
        x = np.random.randn(5, 2)
        point = np.random.randn(5, 2)
        norm = _get_estimate(x, point)
        assert_almost_equal(norm, actual_norm)
