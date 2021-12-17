import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from hyppo.tools import rot_ksamp
from hyppo.ksample import SmoothCFTest


class TestSmoothCF:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue",
        [
            (2000, 1317.0159, 8.134e-277),
            (1000, 546.279, 5.608e-111),
        ],
    )
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2)
        stat, pvalue = SmoothCFTest().test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=1)
        assert_almost_equal(pvalue, obs_pvalue, decimal=100)

    def test_null(self):
        np.random.seed(120)
        x = np.random.randn(500, 10)
        y = np.random.randn(500, 10)
        _, pval = SmoothCFTest().test(x, y)
        assert pval > 0.05

    def test_alternative(self):
        np.random.seed(120)
        x = np.random.randn(500, 10)
        x[:, 1] *= 3
        y = np.random.randn(500, 10)
        _, pval = SmoothCFTest().test(x, y)
        assert pval < 0.05

    @pytest.mark.parametrize(
        "obs_stat, obs_pval",
        [
            (0.2707, 0.8734),
        ],
    )
    def test_one_frequency(self, obs_stat, obs_pval):
        np.random.seed(123456789)
        x = np.random.randn(500, 10)
        y = np.random.randn(500, 10)
        stat, pvalue = SmoothCFTest(num_randfreq=1).test(x, y, random_state=1234)
        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pval, decimal=2)
