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
from ..common import check_categorical, check_2d_array, check_min_samples
import pandas as pd
from ...independence import Dcorr
from ..common import (
    _check_distmat,
    _check_kernmat,
    _multi_check_kernmat,
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
    multi_compute_kern,
    multi_perm_test,
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

    def test_error_threedist(self):
        x = np.arange(10).reshape(-1, 1)
        y = np.arange(10).reshape(1, -1)
        z = np.arange(10).reshape(1, -1)
        assert_raises(ValueError, _check_distmat, x, y, z)

    def test_error_nonzerodiag(self):
        x = np.eye(10, dtype=int) ^ 1  # 0 on diag, 1 elsewhere
        y = np.eye(10, dtype=int) ^ 1  # 0 on diag, 1 elsewhere
        z = np.eye(10, dtype=int)

        assert_raises(ValueError, _check_distmat, x, y, z)

    def test_error_multidistkern(self):
        # raises error if samples are low (< 3)
        x = np.arange(10).reshape(-1, 1)
        y = np.arange(10).reshape(1, -1)
        assert_raises(ValueError, _multi_check_kernmat, x, y)

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
        gamma = 1.0 / (2 * (med**2))

        kernx = pairwise_kernels(x, x, metric="rbf", gamma=gamma)
        kerny = pairwise_kernels(y, y, metric="rbf", gamma=gamma)

        distx, disty = compute_dist(distx, disty, metric=None)
        kernx, kerny = compute_kern(kernx, kerny, metric=None)
        distx_comp, disty_comp = compute_dist(x, y)
        kernx_comp, kerny_comp = compute_kern(x, y)
        kernx_comp1, kerny_comp1 = compute_kern(x, y, metric="rbf")

        assert_array_equal(distx, distx_comp)
        assert_array_equal(disty, disty_comp)
        assert_array_equal(kernx, kernx_comp)
        assert_array_equal(kerny, kerny_comp)
        assert_array_equal(kernx, kerny_comp1)
        assert_array_equal(kerny, kerny_comp1)

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

    def test_multidiskern(self):
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
        gamma = 1.0 / (2 * (med**2))

        kernx = pairwise_kernels(x, x, metric="rbf", gamma=gamma)
        kerny = pairwise_kernels(y, y, metric="rbf", gamma=gamma)

        distx, disty = compute_dist(distx, disty, metric=None)
        kernx, kerny = multi_compute_kern(*(kernx, kerny), metric=None)
        distx_comp, disty_comp = compute_dist(x, y)
        kernx_comp, kerny_comp = multi_compute_kern(*(x, y))
        kernx_comp1, kerny_comp1 = multi_compute_kern(*(x, y), metric="rbf")

        assert_array_equal(distx, distx_comp)
        assert_array_equal(disty, disty_comp)
        assert_array_equal(kernx, kernx_comp)
        assert_array_equal(kerny, kerny_comp)
        assert_array_equal(kernx, kerny_comp1)
        assert_array_equal(kerny, kerny_comp1)

        def gaussian(x, **kwargs):
            return pairwise_kernels(x, x, metric="rbf", **kwargs)

        def euclidean(x, **kwargs):
            return pairwise_distances(x, x, metric="euclidean", **kwargs)

        distx, disty = compute_dist(x, y, metric=euclidean)
        kernx, kerny = multi_compute_kern(*(x, y), metric=gaussian, gamma=gamma)

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

    def test_multipermtest(self):
        x, y = linear(100, 1)

        stat, pvalue, _ = multi_perm_test(Dcorr().statistic, *(x, y))
        assert_almost_equal(stat, 1.0, decimal=1)
        assert_almost_equal(pvalue, 1 / 1000, decimal=1)

    def test_chi2(self):
        x, y = linear(100, 1)

        stat, pvalue = chi2_approx(Dcorr().statistic, x, y)
        assert_almost_equal(stat, 1.0, decimal=1)
        assert_almost_equal(pvalue, 1 / 1000, decimal=1)


class TestSingleMatrixFns:
    """Test utility functions useful for single matrix applications"""

    def setup_method(self):
        """Set up test fixtures before each test method is run"""
        # Create standard test data
        np.random.seed(12345)
        self.n_samples = 100
        self.n_features = 3

        # Create binary treatment
        self.Ts_binary = np.concatenate([np.zeros(50), np.ones(50)])

        # Create multi-category treatment
        self.Ts_multi = np.concatenate([np.zeros(30), np.ones(40), np.ones(30) * 2])

        # Create string categorical treatment
        treatments = ["control", "treatment_A", "treatment_B"]
        self.Ts_strings = np.array(
            [treatments[i % len(treatments)] for i in range(self.n_samples)]
        )

        # Create pandas categorical treatment
        self.Ts_pandas_cat = pd.Series(
            pd.Categorical(
                ["A", "B", "C", "A", "B"] * (self.n_samples // 5),
                categories=["A", "B", "C"],
            )
        )

        # Create feature matrix
        self.Xs = np.random.normal(size=(self.n_samples, self.n_features))

        # Create pandas version of features
        self.Xs_df = pd.DataFrame(
            self.Xs, columns=[f"Feature_{i}" for i in range(self.n_features)]
        )

    def test_check_min_samples(self):
        """Test the check_min_samples function"""
        # Test with arrays having sufficient samples
        check_min_samples(Ts=self.Ts_binary, Xs=self.Xs)

        # Test with minimum samples parameter
        check_min_samples(min_samples=5, Ts=self.Ts_binary, Xs=self.Xs)

        # Test with arrays having too few samples
        small_array1 = np.array([1, 2])
        small_array2 = np.array([3, 4])

        with pytest.raises(
            ValueError, match="has 2 samples, which is below the minimum"
        ):
            check_min_samples(Arr1=small_array1, Arr2=small_array2)

        # Test with inconsistent sample counts
        inconsistent_array = np.ones(self.n_samples + 5)

        with pytest.raises(
            ValueError, match="Inconsistent number of samples across arrays"
        ):
            check_min_samples(Ts=self.Ts_binary, Xs_larger=inconsistent_array)

        # Test with no arrays provided
        with pytest.raises(ValueError, match="No arrays provided for sample checking"):
            check_min_samples()

    def test_check_2d_array(self):
        """Test the check_2d_array function"""
        # Test with 1D array
        arr_1d = np.array([1, 2, 3, 4, 5])
        result_1d = check_2d_array(arr_1d)

        # Should be converted to 2D
        assert result_1d.ndim == 2
        assert result_1d.shape == (5, 1)

        # Test with 2D array
        arr_2d = np.array([[1, 2], [3, 4], [5, 6]])
        result_2d = check_2d_array(arr_2d)

        # Should remain unchanged
        assert result_2d.ndim == 2
        assert result_2d.shape == (3, 2)
        assert np.array_equal(result_2d, arr_2d)

        # Test with 3D array
        arr_3d = np.ones((2, 3, 4))

        with pytest.raises(ValueError, match="Expected a 2-D array"):
            check_2d_array(arr_3d)

    def test_check_categorical(self):
        """Test the check_categorical function with various input types"""
        # Case 1: Simple numeric array
        data_factor, unique_levels, K = check_categorical(self.Ts_binary)
        assert np.array_equal(unique_levels, np.array([0, 1]))
        assert K == 2
        assert np.all(np.isin(data_factor, [0, 1]))

        # Case 2: Numeric array with different data type (float)
        float_array = np.array([1.5, 2.5, 1.5, 3.5])
        data_factor, unique_levels, K = check_categorical(float_array)
        assert np.array_equal(unique_levels, np.array([1.5, 2.5, 3.5]))
        assert K == 3

        # Case 3: String array
        data_factor, unique_levels, K = check_categorical(self.Ts_strings)
        assert np.array_equal(
            unique_levels, np.array(["control", "treatment_A", "treatment_B"])
        )
        assert K == 3
        assert np.all(np.isin(data_factor, [0, 1, 2]))

        # Case 4: pandas Categorical directly
        pd_cat = pd.Categorical(["X", "Y", "Z", "X", "Y"], categories=["X", "Y", "Z"])
        data_factor, unique_levels, K = check_categorical(pd_cat)
        assert np.array_equal(unique_levels, np.array(["X", "Y", "Z"]))
        assert K == 3
        assert np.all(np.isin(data_factor, [0, 1, 2]))

        # Case 5: pandas Series with categorical dtype
        pd_series_cat = pd.Series(pd_cat)
        data_factor, unique_levels, K = check_categorical(pd_series_cat)
        assert np.array_equal(unique_levels, np.array(["X", "Y", "Z"]))
        assert K == 3
        assert np.all(np.isin(data_factor, [0, 1, 2]))

        # Case 6: pandas Series with non-categorical dtype
        pd_series_regular = pd.Series([10, 20, 30, 10, 20])
        data_factor, unique_levels, K = check_categorical(pd_series_regular)
        assert np.array_equal(unique_levels, np.array([10, 20, 30]))
        assert K == 3

        # Case 7: pandas Series with string dtype
        pd_series_str = pd.Series(["apple", "banana", "apple", "cherry"])
        data_factor, unique_levels, K = check_categorical(pd_series_str)
        assert np.array_equal(unique_levels, np.array(["apple", "banana", "cherry"]))
        assert K == 3

        # Case 8: pandas Categorical with ordered=True
        pd_cat_ordered = pd.Categorical(
            ["small", "large", "medium"],
            categories=["small", "medium", "large"],
            ordered=True,
        )
        data_factor, unique_levels, K = check_categorical(pd_cat_ordered)
        assert np.array_equal(unique_levels, np.array(["small", "medium", "large"]))
        assert K == 3

        with pytest.raises(TypeError):
            check_categorical(np.array([]))

        # Case 10: Invalid inputs
        with pytest.raises(TypeError, match="Cannot cast to a categorical vector"):
            check_categorical(np.array([[1, 2], [3, 4]]))  # 2D array

        # Case 11: Non-array input
        with pytest.raises(TypeError, match="Cannot cast to a categorical vector"):
            check_categorical("not_an_array")

    def test_contains_nan(self):
        """Test the contains_nan function"""
        # Create array without NaNs
        array_no_nan = np.array([1.0, 2.0, 3.0])

        # This should not raise an exception
        contains_nan(array_no_nan)

        # Create array with NaNs
        array_with_nan = np.array([1.0, np.nan, 3.0])

        # This should raise an exception
        with pytest.raises(ValueError, match="The input contains nan values"):
            contains_nan(array_with_nan)

        # Test with pandas DataFrame
        df_no_nan = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        contains_nan(df_no_nan)

        df_with_nan = pd.DataFrame({"A": [1, 2, 3], "B": [4, np.nan, 6]})
        with pytest.raises(ValueError, match="The input contains nan values"):
            contains_nan(df_with_nan)

