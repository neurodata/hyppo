import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import power, rot_ksamp
from .. import Hotelling

class TestHHG:
    @pytest.mark.parametrize(
        "n, obs_stat_CM, obs_pvalue_CM, obs_stat_MP",
        [
            (100, -1.25874, 0.25, 4.912e-10),
            (10, -0.359029, 0.25, 0.125874),
        ],
    )
    def test_linear_oned(self, n, obs_stat_CM, obs_pvalue_CM, obs_stat_MP):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2, noise=False)
        CMstat, pvalue = Hotelling(mode ="CM").test(x, y)
        MPstat = Hotelling(mode ="MP").test(x, y)
        
        assert_almost_equal(CMstat, obs_stat_CM, decimal=3)
        assert_almost_equal(pvalue, obs_pvalue_CM, decimal=3)
        assert_almost_equal(MPstat, obs_stat_MP, decimal=3)
    