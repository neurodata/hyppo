import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import cross_corr_ar, nonlinear_process
from .. import LjungBox


class TestLjungBox:
    def test_same_seq(self):
        n = 10
        x = np.arange(1, n + 1).reshape(n, 1)
        y = np.arange(1, n + 1).reshape(n, 1)
        stat = LjungBox(max_lag=1).statistic(x, y)[0]
        assert_almost_equal(stat, 8.06, decimal=2)
