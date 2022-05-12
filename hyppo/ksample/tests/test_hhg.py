import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import rot_ksamp
from .. import HHG


class TestHHG:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue", [(100, 0.515, 4.912e-10), (10, 0.777, 0.125874),],
    )
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2, noise=False)
        stat, pvalue = HHG().test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=3)
        assert_almost_equal(pvalue, obs_pvalue, decimal=3)

    @pytest.mark.parametrize(
        "n", [(100)],
    )
    def test_rep(self, n):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2, noise=False)
        MPstat1 = HHG().statistic(x, y)
        MPstat2 = HHG().statistic(x, y)
        
        assert MPstat1 == MPstat2


class TestHHGTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        rejections = 0
        for i in range(1000):
            x, y = rot_ksamp(
                "multimodal_independence", n=100, p=1, noise=True, degree=90
            )
            stat, pvalue = HHG().test(x, y)
        if pvalue < 0.05:
            rejections += 1
        est_power = rejections / 1000

        assert_almost_equal(est_power, 0, decimal=2)
