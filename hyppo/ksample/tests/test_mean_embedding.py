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
        stat, pvalue = MeanEmbeddingTest(random_state=1234).test(x, y)
        print(stat, pvalue)
        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=10)
