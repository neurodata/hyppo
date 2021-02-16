import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises

from .. import ts_sim


class TestTSSimShape:
    @pytest.mark.parametrize("n", [100, 1000])
    @pytest.mark.parametrize(
        "sim",
        ["indep_ar", "cross_corr_ar", "nonlinear_process"],
    )
    def test_shapes(self, n, sim):
        np.random.seed(123456789)
        x, y = ts_sim(sim, n)
        assert_equal(x.shape, y.shape)


class TestTSSimErrorWarn:
    """Tests errors and warnings."""

    def test_np_inctype(self):
        assert_raises(ValueError, ts_sim, "indep_ar", n="a")

    def test_low_n(self):
        assert_raises(ValueError, ts_sim, "indep_ar", n=1)

    def test_extra_inctype(self):
        assert_raises(ValueError, ts_sim, "indep_ar", n=10, phi="a")

    def test_wrong_sim(self):
        assert_raises(ValueError, ts_sim, "abcd", n=10, phi=1)
