import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import power, rot_ksamp
from .. import HHG

class TestHHG:
    @pytest.mark.parametrize(
        "n, obs_stat_CM, obs_pvalue_CM, obs_stat_MP",
        [
            (100, 0.04, 0.9999988, 4.912e-10),
            (10, 0.3, 0.7869297, 0.125874),
        ],
    )
    def test_linear_oned(self, n, obs_stat_CM, obs_pvalue_CM, obs_stat_MP):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2, noise=False)
        CMstat, pvalue = HHG(mode ="CM").test(x, y)
        MPstat = HHG(mode ="MP").test(x, y)
        
        assert_almost_equal(CMstat, obs_stat_CM, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue_CM, decimal=2)
        assert_almost_equal(MPstat, obs_stat_MP, decimal=2)
        
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue",
        [(100, 8.24e-5, 0.001)],
    )
    def test_rep(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2, noise=False)
        CMstat1, pvalue1 = HHG(mode ="CM").test(x, y)
        CMstat2, pvalue2 = HHG(mode ="CM").test(x, y)
        MPstat1 = HHG(mode ="MP").test(x, y)
        MPstat2 = HHG(mode ="MP").test(x, y)
        
        assert CMstat1 == CMstat2
        assert pvalue1 == pvalue2
        assert MPstat1 == MPstat2
    
class TestHHGTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "HHG",
            sim_type="ksamp",
            sim="multimodal_independence",
            k=2,
            n=100,
            p=1,
            alpha=0.05,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)