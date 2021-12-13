import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from hyppo.tools import rot_ksamp
from hyppo.ksample import MeanEmbeddingTest


class TestMeanEmbedding:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue",
        [
            (2000, 1117.59, 2.064e-239),
            (1000, 345, 1.55e-72),
        ],
    )
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2)
        stat, pvalue = MeanEmbeddingTest().test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=-4)
        assert_almost_equal(pvalue, obs_pvalue, decimal=50)
