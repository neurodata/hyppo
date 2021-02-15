import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises

from .. import SIMULATIONS, gaussian_3samp, ksamp_sim, rot_ksamp


class TestKSampleSimShape:
    @pytest.mark.parametrize("n", [100, 1000])
    @pytest.mark.parametrize("p", [1, 5])
    @pytest.mark.parametrize(
        "indep_sim",
        SIMULATIONS.keys(),
    )
    @pytest.mark.parametrize("k, degree", [(2, 90), (3, [90, 90])])
    def test_shapes(self, indep_sim, k, n, p, degree):
        np.random.seed(123456789)
        sims = rot_ksamp(indep_sim, n, p, k=k, degree=degree)
        sims1 = rot_ksamp(indep_sim, n, p, k=k, degree=degree, pow_type="dim")
        sims2 = ksamp_sim("rot_ksamp", n, sim=indep_sim, p=p, k=k, degree=degree)

        if indep_sim in [
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
            [assert_equal(sim.shape[1], p * 2) for sim in sims]
            [assert_equal(sim.shape[1], p * 2) for sim in sims1]
            [assert_equal(sim.shape[1], p * 2) for sim in sims2]
        else:
            [assert_equal(sim.shape[1], p + 1) for sim in sims]
            [assert_equal(sim.shape[1], p + 1) for sim in sims1]
            [assert_equal(sim.shape[1], p + 1) for sim in sims2]


class TestGaussianSimShape:
    @pytest.mark.parametrize("n", [100, 1000])
    @pytest.mark.parametrize(
        "case",
        [1, 2, 3, 4, 5],
    )
    def test_shapes(self, n, case):
        np.random.seed(123456789)
        sims = gaussian_3samp(n, case=case)

        [assert_equal(sim.shape[0], n) for sim in sims]


class TestKSampleSimErrorWarn:
    """Tests errors and warnings."""

    def test_wrong_powtype(self):
        assert_raises(ValueError, rot_ksamp, sim="linear", n=100, p=1, pow_type="abcd")

    def test_wrong_k(self):
        assert_raises(
            ValueError, rot_ksamp, sim="linear", k=3, n=100, p=1, pow_type="abcd"
        )
        assert_raises(
            ValueError,
            rot_ksamp,
            sim="linear",
            degree=[90, 90],
            n=100,
            p=1,
            pow_type="abcd",
        )
