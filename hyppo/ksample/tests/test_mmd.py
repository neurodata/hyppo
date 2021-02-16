import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import rot_ksamp
from .. import MMD


class TestMMD:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue",
        [(200, 4.28e-7, 0.001), (100, 8.24e-5, 0.001)],
    )
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2)
        stat, pvalue = MMD().test(x, y, auto=False)

        assert_almost_equal(stat, obs_stat, decimal=1)
        assert_almost_equal(pvalue, obs_pvalue, decimal=1)
