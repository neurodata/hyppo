import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises

from .. import condi_indep_sim, COND_SIMULATIONS


"""

class TestTSSimShape:
    @pytest.mark.parametrize("n", [100, 1000])
    @pytest.mark.parametrize(
        "sim",
        ["indep_ar", "cross_corr_ar", "nonlinear_process", "extinct_gaussian_process"],
    )
    def test_shapes(self, n, sim):
        np.random.seed(123456789)
        x, y = ts_sim(sim, n)
        assert_equal(x.shape, y.shape)


class TestTSSimErrorWarn:
    ""Tests errors and warnings."

    def test_np_inctype(self):
        assert_raises(ValueError, ts_sim, "indep_ar", n="a")

    def test_low_n(self):
        assert_raises(ValueError, ts_sim, "indep_ar", n=1)

    def test_extra_inctype(self):
        assert_raises(ValueError, ts_sim, "indep_ar", n=10, phi="a")

    def test_wrong_sim(self):
        assert_raises(ValueError, ts_sim, "abcd", n=10, phi=1)

"""


class TestCondiIndepSimShape:
    @pytest.mark.parametrize("n", [100, 1000])
    @pytest.mark.parametrize("p", [1, 5])
    @pytest.mark.parametrize(
        "sim",
        COND_SIMULATIONS.keys(),
    )
    def test_shapes(self, n, p, sim):
        np.random.seed(123456789)
        x, y, z = condi_indep_sim(sim, n, p)
        nx, px = x.shape
        ny, py = y.shape
        nz, pz = z.shape

        n = np.array([nx, ny, nz])

        assert np.all(n == nx)


class TestCondiIndepSimErrorWarn:
    """Tests errors and warnings."""

    def test_np_inctype(self):
        assert_raises(ValueError, condi_indep_sim, n="a", p=1, sim="independent_normal")
        assert_raises(
            ValueError, condi_indep_sim, n=10, p=7.0, sim="independent_normal"
        )

    def test_low_n(self):
        assert_raises(ValueError, condi_indep_sim, n=3, p=1, sim="independent_normal")

    def test_low_p(self):
        assert_raises(ValueError, condi_indep_sim, n=5, p=0, sim="independent_normal")

    def test_wrong_sim(self):
        assert_raises(ValueError, condi_indep_sim, n=100, p=1, sim="abcd")
