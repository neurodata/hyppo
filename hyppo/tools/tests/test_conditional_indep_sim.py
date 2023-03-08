import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises

from .. import SIMULATIONS, indep_sim


class TestIndepSimShape:
    @pytest.mark.parametrize("n", [100, 1000])
    @pytest.mark.parametrize("p", [1, 5])
    @pytest.mark.parametrize(
        "sim",
        SIMULATIONS.keys(),
    )
    def test_shapes(self, n, p, sim):
        np.random.seed(123456789)
        x, y = SIMULATIONS[sim](n, p)
        x1, y1 = indep_sim(sim, n, p)
        nx, px = x.shape
        ny, py = y.shape
        nx1, px1 = x1.shape
        ny1, py1 = y1.shape

        if sim in [
            "joint_normal",
            "logarithmic",
            "sin_four_pi",
            "sin_sixteen_pi",
            "two_parabolas",
            "square",
            "diamond",
            "circle",
            "ellipse",
            "multiplicative_noise",
            "multimodal_independence",
        ]:
            assert_equal(px, py)
            assert_equal(px1, py1)
        else:
            assert_equal(px, py * p)
            assert_equal(px1, py1 * p)
        assert_equal(nx, ny)
        assert_equal(nx1, ny1)


class TestIndepSimErrorWarn:
    """Tests errors and warnings."""

    def test_np_inctype(self):
        assert_raises(ValueError, indep_sim, n="a", p=1, sim="linear")
        assert_raises(ValueError, indep_sim, n=10, p=7.0, sim="linear")

    def test_low_n(self):
        assert_raises(ValueError, indep_sim, n=3, p=1, sim="linear")

    def test_low_p(self):
        assert_raises(ValueError, indep_sim, n=5, p=0, sim="linear")

    def test_extra_inctype(self):
        assert_raises(ValueError, indep_sim, n=5, p=0, sim="linear", low="a")

    def test_wrong_sim(self):
        assert_raises(ValueError, indep_sim, n=100, p=1, sim="abcd")

    def test_joint_lowcov(self):
        assert_raises(ValueError, indep_sim, n=100, p=20, sim="joint_normal")
