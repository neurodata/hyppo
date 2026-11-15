import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_raises,
    assert_warns,
)

from ...tools import linear, logarithmic, power
from .. import KCI


@pytest.mark.filterwarnings("ignore:No conditioning variable z:UserWarning")
class TestKCI:
    @pytest.mark.parametrize("n", [100])
    @pytest.mark.parametrize("obs_stat", [544.691])
    @pytest.mark.parametrize("obs_pvalue", [0])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        stat1, pvalue1 = KCI().test(x, y)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(pvalue1, obs_pvalue, decimal=2)

    @pytest.mark.parametrize("n", [100])
    @pytest.mark.parametrize("obs_stat", [88.526])
    @pytest.mark.parametrize("obs_pvalue", [.00027])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = logarithmic(n, 1)
        stat1, pvalue1 = KCI().test(x, y)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(pvalue1, obs_pvalue, decimal=2)

    @pytest.mark.parametrize("n", [100, 200])
    def test_rep(self, n):
        x, y = linear(n, 1)
        stat1, pvalue1 = KCI().test(x, y)
        stat2, pvalue2 = KCI().test(x, y)

        assert stat1 == stat2
        assert pvalue1 == pvalue2


def _chain(n, seed):
    # x and y both depend on z, and are independent given z
    rng = np.random.RandomState(seed)
    z = rng.normal(size=(n, 1))
    x = z + 0.5 * rng.normal(size=(n, 1))
    y = z**2 + 0.5 * rng.normal(size=(n, 1))
    return x, y, z


class TestKCIConditional:
    @pytest.mark.parametrize("n", [100, 300])
    def test_conditionally_independent(self, n):
        x, y, z = _chain(n, 0)
        stat, pvalue = KCI().test(x, y, z)

        assert pvalue > 0.05
        assert_almost_equal(stat, KCI().statistic(x, y, z))

    @pytest.mark.parametrize("n", [100, 300])
    def test_conditionally_dependent(self, n):
        x, y, z = _chain(n, 0)
        y = y + x  # y now depends on x also given z
        stat, pvalue = KCI().test(x, y, z)

        assert pvalue < 0.05

    def test_z_is_used(self):
        # x and y are independent given the true z, and not given a z that
        # has been shuffled so that it no longer carries the dependence
        x, y, z = _chain(300, 1)
        z_shuffled = z[np.random.RandomState(5).permutation(300)]

        assert KCI().test(x, y, z).pvalue > 0.05
        assert KCI().test(x, y, z_shuffled).pvalue < 0.05

    @pytest.mark.filterwarnings("ignore:No conditioning variable z:UserWarning")
    def test_z_changes_statistic(self):
        x, y, z = _chain(100, 2)
        assert KCI().statistic(x, y, z) != KCI().statistic(x, y)
        assert KCI().statistic(x, y, z) != KCI().statistic(x, y, z[::-1])

    def test_missing_z_warns(self):
        x, y, _ = _chain(100, 3)
        with pytest.warns(UserWarning):
            KCI().test(x, y)

    def test_missing_z_warns_statistic(self):
        x, y, _ = _chain(100, 3)
        with pytest.warns(UserWarning):
            KCI().statistic(x, y)

    @pytest.mark.parametrize(
        "n, stat, pvalue",
        [(100, 0.99449337, 0.17610331), (300, 23.0644991, 0.10251098)],
    )
    def test_matches_causal_learn(self, n, stat, pvalue):
        # reference values: causal-learn 0.1.4.8, KCI_CInd().compute_pvalue(x, y, z)
        # on the same data (equal to about 1e-11)
        x, y, z = _chain(n, 0)
        out = KCI().test(x, y, z)

        assert_allclose(out.stat, stat, rtol=1e-6)
        assert_allclose(out.pvalue, pvalue, rtol=1e-6)

    def test_matches_causal_learn_multivariate_z(self):
        # causal-learn 0.1.4.8, KCI_CInd().compute_pvalue(x, y, z), 3-D z
        rng = np.random.RandomState(7)
        z = rng.normal(size=(120, 3))
        s = z.sum(axis=1, keepdims=True)
        x = s + 0.5 * rng.normal(size=(120, 1))
        y = s**2 + 0.5 * rng.normal(size=(120, 1))
        out = KCI().test(x, y, z)

        assert_allclose(out.stat, 0.00373762842, rtol=1e-6)
        assert_allclose(out.pvalue, 0.631778898, rtol=1e-6)

    def test_z_row_mismatch(self):
        x, y, z = _chain(100, 5)
        with pytest.raises(ValueError):
            KCI().test(x, y, z[:-5])
        with pytest.raises(ValueError):
            KCI().statistic(x, y, z[:-5])

    def test_z_nan(self):
        x, y, z = _chain(100, 5)
        z[3] = np.nan
        with pytest.raises(ValueError):
            KCI().test(x, y, z)

    def test_rep(self):
        x, y, z = _chain(100, 4)
        assert KCI().test(x, y, z) == KCI().test(x, y, z)
