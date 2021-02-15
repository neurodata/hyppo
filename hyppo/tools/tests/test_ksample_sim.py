import numpy as np
import pytest
from numpy.testing import assert_equal

from .. import SIMULATIONS, rot_ksamp


class TestKSampleShape:
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
        else:
            [assert_equal(sim.shape[1], p + 1) for sim in sims]
