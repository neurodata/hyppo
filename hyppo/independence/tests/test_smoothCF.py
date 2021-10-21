import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_approx_equal

from ...tools import linear, multimodal_independence, power, spiral
from .. import MGC


class SmoothCF(object):
    """Test validity of SmoothCF test statistic"""



class SmoothCFError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "MGC",
            sim_type="indep",
            sim="multimodal_independence",
            n=50,
            p=1,
            alpha=0.05,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)
