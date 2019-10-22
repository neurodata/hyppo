import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from ...benchmarks.ksample_sim import linear_2samp
from .. import UnpairKSample
from ...independence import CannCorr, Dcorr


class TestUnpairKSamp:
    @pytest.mark.parametrize("n, obs_stat, obs_pvalue, indep_test", [
        (10, 0.0162, 0.693, CannCorr),
        (100, 8.24e-5, 0.981, CannCorr),
        (1000, 4.28e-7, 1.0, CannCorr),
        (10, 0.153, 0.091, Dcorr),
        (50, 0.0413, 0.819, Dcorr),
        (100, 0.0237, 0.296, Dcorr)
    ])
    def test_twosamp_linear_oned(self, n, obs_stat, obs_pvalue, indep_test):
        np.random.seed(123456789)
        x, y = linear_2samp(n, 1, noise=0)
        stat, pvalue = UnpairKSample(indep_test).test([x, y])

        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(pvalue, obs_pvalue, decimal=2)

