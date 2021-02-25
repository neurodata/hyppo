import numpy as np
from numpy.testing import assert_almost_equal, assert_raises

from .. import power


class TestPower:
    def test_power_perm(self):
        np.random.seed(123456789)
        est_power = power("RV", "indep", sim="linear", p=1)
        assert_almost_equal(est_power, 1.0, decimal=1)

    def test_power_fast(self):
        np.random.seed(123456789)
        est_power = power("Dcorr", "indep", sim="linear", p=1, auto=True)
        assert_almost_equal(est_power, 1.0, decimal=1)

    def test_power_fast_nless20(self):
        np.random.seed(123456789)
        est_power = power("Dcorr", "indep", sim="linear", n=19, p=1, auto=True)
        assert_almost_equal(est_power, np.nan, decimal=1)

    def test_ksamp(self):
        np.random.seed(123456789)
        est_power = power("Dcorr", "ksamp", sim="linear", p=1, auto=True)
        assert_almost_equal(est_power, 1.0, decimal=1)

    def test_gaussian(self):
        np.random.seed(123456789)
        est_power = power("Dcorr", "gauss", auto=True, case=2)
        assert_almost_equal(est_power, 1.0, decimal=1)


class TestPowerErrorWarn:
    def test_power_nosim(self):
        np.random.seed(123456789)
        assert_raises(ValueError, power, "CCA", "abcd")

    def test_power_noindep(self):
        np.random.seed(123456789)
        assert_raises(ValueError, power, "abcd", "indep")

    def test_power_nomaxmargin(self):
        np.random.seed(123456789)
        assert_raises(ValueError, power, ["MaxMargin", "abcd"], "indep")
