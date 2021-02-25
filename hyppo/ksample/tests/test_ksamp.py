import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises

from ...tools import power, rot_ksamp
from .. import KSample


class TestKSample:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue, indep_test",
        [(1000, 4.28e-7, 1.0, "CCA"), (100, 8.24e-5, 0.001, "Dcorr")],
    )
    def test_twosamp_linear_oned(self, n, obs_stat, obs_pvalue, indep_test):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2)
        stat, pvalue = KSample(indep_test).test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=1)
        assert_almost_equal(pvalue, obs_pvalue, decimal=1)

    def test_rf(self):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", 50, 1, k=2)
        stat, _, _ = KSample("KMERF").test(x, y, reps=0)

        assert_almost_equal(stat, 0.2714, decimal=1)

    def test_maxmargin(self):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", 50, 1, k=2)
        stat, _ = KSample(["MaxMargin", "Dcorr"]).test(x, y, reps=0)

        assert_almost_equal(stat, 0.0317, decimal=1)


class TestKSampleErrorWarn:
    """Tests errors and warnings derived from MGC."""

    def test_no_indeptest(self):
        # raises error if not indep test
        assert_raises(ValueError, KSample, "abcd")

    def test_no_second_maxmargin(self):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", 50, 1, k=2)
        assert_raises(ValueError, KSample, ["MaxMargin", "abcd"])


class TestKSampleTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "CCA",
            sim_type="ksamp",
            sim="multimodal_independence",
            k=2,
            n=100,
            p=1,
            alpha=0.05,
            auto=True,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)
