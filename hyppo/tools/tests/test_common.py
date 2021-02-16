import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_array_less,
    assert_equal,
    assert_raises,
    assert_warns,
)
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels

from ...independence import Dcorr
from ..common import (
    _check_distmat,
    _check_kernmat,
    _PermTree,
    check_ndarray_xy,
    check_perm_blocks,
    check_perm_blocks_dim,
    check_reps,
    chi2_approx,
    compute_dist,
    compute_kern,
    contains_nan,
    convert_xy_float64,
    perm_test,
)
from ..indep_sim import linear


class TestPermTree:
    """
    Tests that permutations are restricted correctly
    """

    def test_within_permutatins(self):
        np.random.seed(1)
        # i.e. case: y = np.asarray([0,1,0,1,0,1])
        blocks = np.vstack(
            (
                [-1, 1],
                [-1, 2],
                [-2, 1],
                [-2, 2],
                [-3, 1],
                [-3, 2],
            )
        )
        perm_tree = _PermTree(blocks)
        original_indices = perm_tree.original_indices()
        perms = np.asarray([perm_tree.permute_indices() for _ in range(10)])
        assert_array_less(np.abs(original_indices - perms), 2)
        assert_allclose(
            np.mean(perms, axis=0), [0.5, 0.5, 2.5, 2.5, 4.5, 4.5], rtol=0, atol=0.2
        )

    def test_across_permutations(self):
        np.random.seed(0)
        # i.e. case: y = np.asarray([0,0,1,1,2,2])
        blocks = np.vstack(
            (
                [1, -1],
                [1, -2],
                [2, -1],
                [2, -2],
                [3, -1],
                [3, -2],
            )
        )
        perm_tree = _PermTree(blocks)
        _ = perm_tree.original_indices()
        perms = np.asarray([perm_tree.permute_indices() for _ in range(100)])
        assert_equal(perms[0][1::2] - perms[0][::2], [1, 1, 1])
        assert_allclose(np.mean(perms, axis=0), [2, 3, 2, 3, 2, 3], rtol=0, atol=0.2)

    def test_fixed_permutation(self):
        np.random.seed(0)
        blocks = [-1, -2, -3, -4]
        perm_tree = _PermTree(blocks)
        assert_equal(perm_tree.permute_indices(), perm_tree.original_indices())

    def test_semifixed(self):
        np.random.seed(1)
        blocks = [1, 2, -3, -4]
        perm_tree = _PermTree(blocks)
        perms = np.asarray([perm_tree.permute_indices() for _ in range(10)])
        assert_equal(perms[0][2:], perm_tree.original_indices()[2:])
        assert_allclose(np.mean(perms, axis=0)[:2], [0.5, 0.5], rtol=0, atol=0.2)

    def test_non_int_inputs(self):
        blocks = ["a", "b", "c"]
        _ = _PermTree(blocks)

    def test_block_sizes(self):
        blocks = [1, 1, 2]
        assert_raises(ValueError, _PermTree, blocks)

    def test_noblock(self):
        perm_blocks = check_perm_blocks(None)
        assert_equal(perm_blocks, None)

    def test_not_ndarray(self):
        perm_blocks = (1, 2, 3)
        assert_raises(TypeError, check_perm_blocks, perm_blocks)

    def test_incorrect_dim(self):
        perm_blocks = np.arange(125).reshape(5, 5, 5)
        assert_raises(ValueError, check_perm_blocks, perm_blocks)

    def test_permblock_dim(self):
        perm_blocks = np.arange(100).reshape(50, 2)
        y = np.arange(100).reshape(10, 10)
        assert_raises(ValueError, check_perm_blocks_dim, perm_blocks, y)


class TestErrorWarn:
    """Tests errors and warnings."""

    def test_error_notndarray(self):
        # raises error if x or y is not a ndarray
        x = np.arange(20)
        y = [5] * 20
        assert_raises(TypeError, check_ndarray_xy, x, y)
        assert_raises(TypeError, check_ndarray_xy, y, x)

    def test_conv_float(self):
        # raises error if number of samples different (n)
        x = np.arange(20)
        y = np.arange(20)
        x, y = convert_xy_float64(x, y)
        assert_equal(x.dtype, np.float64)
        assert_equal(y.dtype, np.float64)

    def test_error_distkern(self):
        # raises error if samples are low (< 3)
        x = np.arange(10).reshape(-1, 1)
        y = np.arange(10).reshape(1, -1)
        assert_raises(ValueError, _check_distmat, x, y)
        assert_raises(ValueError, _check_kernmat, x, y)

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        x = np.arange(20, dtype=float)
        x[0] = np.nan
        assert_raises(ValueError, contains_nan, x)

    @pytest.mark.parametrize(
        "reps", [-1, "1"]  # reps is negative  # reps is not integer
    )
    def test_error_reps(self, reps):
        # raises error if reps is negative
        assert_raises(ValueError, check_reps, reps)

    def test_warn_reps(self):
        # raises error if reps is negative
        reps = 100
        assert_warns(RuntimeWarning, check_reps, reps)


class TestHelper:
    """Tests errors and warnings derived from MGC."""

    def test_diskern(self):
        np.random.seed(123456789)
        x, y = linear(100, 1)
        distx = pairwise_distances(x, x)
        disty = pairwise_distances(y, y)

        l1 = pairwise_distances(x, metric="l1")
        n = l1.shape[0]
        med = np.median(
            np.lib.stride_tricks.as_strided(
                l1, (n - 1, n + 1), (l1.itemsize * (n + 1), l1.itemsize)
            )[:, 1:]
        )
        gamma = 1.0 / (2 * (med ** 2))

        kernx = pairwise_kernels(x, x, metric="rbf", gamma=gamma)
        kerny = pairwise_kernels(x, x, metric="rbf", gamma=gamma)

        distx, disty = compute_dist(distx, disty, metric=None)
        kernx, kerny = compute_kern(kernx, kerny, metric=None)
        distx_comp, disty_comp = compute_dist(x, y)
        kernx_comp, kerny_comp = compute_kern(x, y)

        assert_array_equal(distx, distx_comp)
        assert_array_equal(disty, disty_comp)
        assert_array_equal(kernx, kernx_comp)
        assert_array_equal(kerny, kerny_comp)

        def gaussian(x, **kwargs):
            return pairwise_kernels(x, x, metric="rbf", **kwargs)

        def euclidean(x, **kwargs):
            return pairwise_distances(x, x, metric="euclidean", **kwargs)

        distx, disty = compute_dist(x, y, metric=euclidean)
        kernx, kerny = compute_kern(x, y, metric=gaussian, gamma=gamma)

        assert_array_equal(distx, distx_comp)
        assert_array_equal(disty, disty_comp)
        assert_array_equal(kernx, kernx_comp)
        assert_array_equal(kerny, kerny_comp)

    def test_permtest(self):
        x, y = linear(100, 1)

        stat, pvalue, _ = perm_test(Dcorr().statistic, x, y, is_distsim=False)
        assert_almost_equal(stat, 1.0, decimal=1)
        assert_almost_equal(pvalue, 1 / 1000, decimal=1)

        x = pairwise_distances(x, x)
        y = pairwise_distances(y, y)
        stat, pvalue, _ = perm_test(Dcorr().statistic, x, y, is_distsim=True)
        assert_almost_equal(stat, 1.0, decimal=1)
        assert_almost_equal(pvalue, 1 / 1000, decimal=1)

    def test_chi2(self):
        x, y = linear(100, 1)

        stat, pvalue = chi2_approx(Dcorr().statistic, x, y)
        assert_almost_equal(stat, 1.0, decimal=1)
        assert_almost_equal(pvalue, 1 / 1000, decimal=1)
