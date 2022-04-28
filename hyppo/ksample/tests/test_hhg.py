import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import power, rot_ksamp
from .. import HHG


class TestHHG:
    @pytest.mark.parametrize(
        "n, obs_stat_MP", [(100, 4.912e-10), (10, 0.125874),],
    )
    def test_linear_oned(self, n, obs_stat_MP):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2, noise=False)
        MPstat = HHG().test(x, y)

        assert_almost_equal(MPstat, obs_stat_MP, decimal=2)

    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue", [(100, 8.24e-5, 0.001)],
    )
    def test_rep(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2, noise=False)
        MPstat1 = HHG(mode="MP").test(x, y)
        MPstat2 = HHG(mode="MP").test(x, y)

        assert MPstat1 == MPstat2


class TestHHGTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        rejections = 0
        for i in range(1000):
            x, y = rot_ksamp(
                "multimodal_independence", n=100, p=1, noise=True, degree=90
            )
            MPstat = HHG().test(x, y)
        if MPstat < 0.05:
            rejections += 1
        est_power = rejections / 1000

        assert_almost_equal(est_power, 0, decimal=3)
