import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from .. import DcorrX


class TestDcorrXStat:
    def test_zero_var(self):
        x = np.ones(4)
        y = np.arange(4)
        stat, _ = DcorrX().test(x, y)
        assert_almost_equal(stat, 0.0)

    def test_multiple_lags(self):
        x = np.ones(6)
        y = np.arange(1, 7)
        stat, _ = DcorrX(max_lag=3).test(x, y)
        assert_almost_equal(stat, 0.0)
