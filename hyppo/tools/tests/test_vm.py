import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises
import warnings
from scipy.stats import gaussian_kde
import pandas as pd

from .. import CATE_SIMULATIONS, cate_sim, simulate_covars
from ..common import check_min_samples, check_2d_array, check_categorical, contains_nan
from .. import VectorMatch, _CleanInputsVM


class TestCommonFunctions:
    """Test the utility functions that were refactored out of _CleanInputsVM"""

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


class TestCleanInputsVM:
    """Test the input cleaning and validation directly"""

    def setup_method(self):
        """Set up test fixtures before each test method is run"""
        # Create standard test data
        np.random.seed(123456789)
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

    def test_1d_feature_conversion(self):
        """Test that 1D features are properly converted to 2D"""
        # Create 1D feature array
        Xs_1d = np.random.normal(size=self.n_samples)

        # Test with _CleanInputsVM directly
        cleaner = _CleanInputsVM(self.Ts_binary, Xs_1d)
        assert cleaner.Xs_df.shape == (self.n_samples, 1)

    def test_nan_detection_covariates(self):
        """Test that NaN values in Xs are detected and proper error is raised"""
        # Create features with NaN
        Xs_with_nan = self.Xs.copy()
        Xs_with_nan[10, 0] = np.nan

        # Test with _CleanInputsVM directly
        with pytest.raises(
            ValueError,
            match="Error checking `Xs'. Error: The input contains nan values",
        ):
            cleaner = _CleanInputsVM(self.Ts_binary, Xs_with_nan)

    def test_nan_detection_in_treatments(self):
        """Test that NaN values in Ts are detected and proper error is raised"""
        # Create treatments with NaN
        Ts_with_nan = self.Ts_binary.copy()
        Ts_with_nan[5] = np.nan

        # Test with _CleanInputsVM directly
        with pytest.raises(
            ValueError,
            match="Error checking `Ts'. Error: The input contains nan values",
        ):
            cleaner = _CleanInputsVM(Ts_with_nan, self.Xs)

    def test_ts_categorical_conversion(self):
        """Test that treatments are properly converted to categorical"""
        # Test binary treatment
        cleaner = _CleanInputsVM(self.Ts_binary, self.Xs)
        assert_equal(cleaner.unique_treatments, np.array([0, 1]))
        assert cleaner.K == 2

        # Test multi-category treatment
        cleaner = _CleanInputsVM(self.Ts_multi, self.Xs)
        assert_equal(cleaner.unique_treatments, np.array([0, 1, 2]))
        assert cleaner.K == 3

        # Test string categorical treatments
        cleaner = _CleanInputsVM(self.Ts_strings, self.Xs)
        assert_equal(
            cleaner.unique_treatments,
            np.array(["control", "treatment_A", "treatment_B"]),
        )
        assert cleaner.K == 3
        # Check that string treatments are properly encoded as numbers
        assert np.all(np.isin(cleaner.Ts_factor, [0, 1, 2]))

        # Test pandas categorical Series
        cleaner = _CleanInputsVM(self.Ts_pandas_cat, self.Xs)
        assert_equal(cleaner.unique_treatments, np.array(["A", "B", "C"]))
        assert cleaner.K == 3

    def test_propensity_formula_generation(self):
        """Test formula generation for propensity model"""
        # Test with default formula
        cleaner = _CleanInputsVM(self.Ts_binary, self.Xs_df)
        assert cleaner.formula.startswith("Ts ~")
        assert "Feature_0" in cleaner.formula

        # Test with custom formula
        custom_formula = "Feature_0 + np.log(Feature_1)"
        cleaner = _CleanInputsVM(
            self.Ts_binary, self.Xs_df, prop_form_rhs=custom_formula
        )
        assert cleaner.formula == f"Ts ~ {custom_formula}"

    def test_design_matrix_construction(self):
        """Test that the design matrix correctly reflects the formula"""
        # Test with default formula (all features included)
        cleaner = _CleanInputsVM(self.Ts_binary, self.Xs_df)

        # The design matrix should have an intercept column + all feature columns
        expected_cols = self.n_features + 1  # +1 for intercept
        assert cleaner.Xs_design.shape[1] == expected_cols

        # First column should be the intercept (all ones)
        assert np.allclose(cleaner.Xs_design.iloc[:, 0], 1.0)

        # Test with custom formula including transformations
        # Create simple dataframe with positive values (for log transformation)
        Xs_positive = pd.DataFrame(
            {
                "A": np.random.uniform(0.1, 10, size=self.n_samples),
                "B": np.random.uniform(0.1, 10, size=self.n_samples),
                "C": np.random.uniform(0.1, 10, size=self.n_samples),
            }
        )

        # Use formula with math transformations
        custom_formula = "A + np.log(B) + np.sqrt(C)"
        cleaner = _CleanInputsVM(
            self.Ts_binary, Xs_positive, prop_form_rhs=custom_formula
        )

        # Should have 4 columns: intercept, A, log(B), sqrt(C)
        assert cleaner.Xs_design.shape[1] == 4

        # Verify the transformed columns match expected values
        # Column 1 should be raw A values
        assert np.allclose(cleaner.Xs_design.iloc[:, 1], Xs_positive["A"])

        # Column 2 should be log(B) values
        assert np.allclose(cleaner.Xs_design.iloc[:, 2], np.log(Xs_positive["B"]))

        # Column 3 should be sqrt(C) values
        assert np.allclose(cleaner.Xs_design.iloc[:, 3], np.sqrt(Xs_positive["C"]))

    def test_sample_size_validation(self):
        """Test minimum sample size validation"""
        # Create small datasets
        Ts_small = np.array([0, 1])
        Xs_small = np.random.normal(size=(2, 2))

        # Test with _CleanInputsVM directly
        with pytest.raises(ValueError, match="below the minimum of"):
            cleaner = _CleanInputsVM(Ts_small, Xs_small)

    def test_sample_count_mismatch(self):
        """Test detection of sample count mismatch between Ts and Xs"""
        # Create mismatched datasets
        Ts_short = self.Ts_binary[:-5]  # 5 samples fewer than Xs

        # Test with _CleanInputsVM directly
        with pytest.raises(ValueError, match="Inconsistent number of samples"):
            cleaner = _CleanInputsVM(Ts_short, self.Xs)

    def test_custom_formula_with_numpy_array(self):
        """Test error when providing custom formula with numpy array features"""
        # Test with custom formula but numpy array features
        custom_formula = "X0 + X1 + X2"

        # Should raise error because can't use custom formula with non-DataFrame
        with pytest.raises(ValueError, match="propensity formula"):
            cleaner = _CleanInputsVM(
                self.Ts_binary, self.Xs, prop_form_rhs=custom_formula
            )

    def test_treatment_factor_encoding(self):
        """Test that treatments are properly encoded as factors"""
        # Test with string treatments
        cleaner = _CleanInputsVM(self.Ts_strings, self.Xs)

        # Verify treatment encoding is consistent
        unique_treatments = np.unique(self.Ts_strings)
        for i, treatment in enumerate(unique_treatments):
            # Find all indices where original treatment is this value
            original_indices = np.where(self.Ts_strings == treatment)[0]
            # Check that all those indices have the same encoded value in Ts_factor
            encoded_values = cleaner.Ts_factor[original_indices]
            assert np.all(encoded_values == encoded_values[0])

        # Test with pandas categorical
        cleaner = _CleanInputsVM(self.Ts_pandas_cat, self.Xs)

        # The encoding should preserve the order of categories in the pandas Categorical
        for i, cat in enumerate(self.Ts_pandas_cat.cat.categories):
            # Find indices where original category is this value
            original_indices = np.where(self.Ts_pandas_cat == cat)[0]
            # Check that encoded value matches the category index
            assert np.all(cleaner.Ts_factor[original_indices] == i)

    def test_categorical_covariate_design_matrix(self):
        """Test that categorical covariates are properly encoded in the design matrix."""
        # Create a feature matrix with mixed types
        numeric_feature = np.random.normal(size=self.n_samples)
        categorical_feature = np.random.choice(
            ["red", "green", "blue"], size=self.n_samples
        )

        # Convert to DataFrame
        mixed_df = pd.DataFrame(
            {"numeric": numeric_feature, "color": categorical_feature}
        )

        # Initialize with these features
        cleaner = _CleanInputsVM(self.Ts_binary, mixed_df)

        # Check the design matrix shape - should have columns for:
        # intercept + numeric + (3-1) dummy variables for color
        expected_cols = 1 + 1 + (3 - 1)  # intercept + numeric + (categories-1)
        assert cleaner.Xs_design.shape[1] == expected_cols

        # Check column names - should include "color[T.green]" and "color[T.red]" or similar
        # (exact format may vary based on patsy's treatment coding)
        column_names = cleaner.Xs_design.columns.tolist()

        # One column should be the intercept
        assert "Intercept" in column_names

        # One column should be the numeric feature
        assert "numeric" in column_names or any(
            "numeric" in col for col in column_names
        )

        # There should be dummy variable columns for the categorical feature
        assert any("color" in col for col in column_names)

        # Verify that the categorical feature is being treated as categorical
        # by checking for dummy variable patterns in the design matrix
        categorical_cols = [col for col in column_names if "color" in col]
        assert len(categorical_cols) == 2  # Should be 2 dummies for 3 categories

        # Check that the dummy variables contain only 0s and 1s
        for col in categorical_cols:
            values = cleaner.Xs_design[col].unique()
            assert set(values).issubset(
                {0, 1}
            ), f"Column {col} contains values other than 0 and 1"


def approx_overlap(X1, X2, nbreaks=100):
    """
    Calculate approximate overlap between two distributions using KDE.
    """
    xbreaks = np.linspace(-1, 1, nbreaks)
    x1_dens = gaussian_kde(X1.reshape(-1))(xbreaks)
    x2_dens = gaussian_kde(X2.reshape(-1))(xbreaks)

    # Normalize densities
    x1_dens = x1_dens / np.sum(x1_dens)
    x2_dens = x2_dens / np.sum(x2_dens)

    # Calculate overlap
    return np.sum(np.minimum(x1_dens, x2_dens))


class TestVectorMatch:
    def test_initialization(self):
        """Test initialization and that things are what we expect."""
        vm = VectorMatch(retain_ratio=0.1)
        assert vm.retain_ratio == 0.1
        assert vm.is_fitted is False
        assert vm.balanced_ids is None
        assert vm.model is None

    def test_vector_matching_with_one_oddball_per_group(self):
        """Test vector matching correctly excludes outliers."""
        nrep = 100
        results = []

        # Set seed for reproducibility
        rng = np.random.RandomState(123456789)

        for i in range(nrep):
            Ts = np.concatenate([np.ones(100), np.ones(100) * 2])

            # Create features with outliers at positions 0 and 199
            X1 = np.concatenate([[-4], rng.uniform(size=198), [5]])
            X2 = np.concatenate([[-4], rng.uniform(size=198), [5]])
            X3 = rng.uniform(size=200)
            Xs = np.column_stack([X1, X2, X3])

            # Suppress warnings during test execution
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                retained_ids = VectorMatch().fit(Ts, Xs)

            # Check if samples 0 and 199 are excluded
            excl_sample_0 = 0 not in retained_ids
            excl_sample_199 = 199 not in retained_ids

            # Check if most other samples are included (>80%)
            included_samples = (
                sum([i in retained_ids for i in range(1, 199)]) / 198 > 0.8
            )

            # Success if both outliers excluded and most normal samples included
            results.append(excl_sample_0 and excl_sample_199 and included_samples)

        # Test passes if this works most of the time (>80%)
        assert np.mean(results) > 0.8

    def test_unbalancedness_affects_sample_retention(self):
        """Test that as unbalancedness increases, fewer samples are retained by VM."""
        nrep = 20
        results = []

        # Set seed for reproducibility
        rng = np.random.RandomState(123456789)

        # Generate 20 unique seeds
        seeds = rng.randint(0, 1000000, size=20)

        for i in range(nrep):
            # Create random state for this iteration
            rngi = np.random.RandomState(seeds[i])

            # High balance (more samples retained)
            sim_high = cate_sim("Sigmoidal", n=300, p=3, balance=1.0, random_state=rngi)
            retained_high = VectorMatch().fit(sim_high["Ts"], sim_high["Xs"])

            # Moderate balance
            sim_mod = cate_sim("Sigmoidal", n=300, p=3, balance=0.7, random_state=rngi)
            retained_mod = VectorMatch().fit(sim_mod["Ts"], sim_mod["Xs"])

            # Low balance (fewer samples retained)
            sim_low = cate_sim("Sigmoidal", n=300, p=3, balance=0.4, random_state=rngi)
            retained_low = VectorMatch().fit(sim_low["Ts"], sim_low["Xs"])

            # Check if the number of retained samples decreases as unbalancedness increases
            lengths = [len(retained_high), len(retained_mod), len(retained_low)]
            ranks = np.argsort(np.argsort(lengths))

            # Test passes if the ranks are as expected (3,2,1)
            results.append(np.array_equal(ranks, [2, 1, 0]))

        assert np.mean(results) > 0.8

    def test_warning_when_samples_retained_is_low(self):
        """Test VM throws warning when samples retained is low."""
        rng = np.random.RandomState(123456789)
        sim_low = cate_sim("Sigmoidal", n=200, p=3, balance=0.2, random_state=rng)

        with pytest.warns(UserWarning):
            VectorMatch(retain_ratio=0.9).fit(sim_low["Ts"], sim_low["Xs"])

    def test_error_when_no_samples_retained(self):
        """Test VM throws error when no samples are retained."""
        rng = np.random.RandomState(123456789)
        sim_low = cate_sim(
            "Sigmoidal", n=200, p=3, balance=0, alpha=10, beta=1, random_state=rng
        )

        with pytest.raises(ValueError):
            VectorMatch(retain_ratio=0).fit(sim_low["Ts"], sim_low["Xs"])

    def test_is_fitted_flag(self):
        """Test that is_fitted flag is properly set after fitting."""
        rng = np.random.RandomState(123456789)
        sim = cate_sim("Sigmoidal", n=200, p=1, balance=0.5, random_state=rng)

        vm = VectorMatch()
        assert vm.is_fitted is False

        vm.fit(sim["Ts"], sim["Xs"])
        assert vm.is_fitted is True

        # Test that attempting to fit again raises an error
        with pytest.raises(ValueError, match="already been fit"):
            vm.fit(sim["Ts"], sim["Xs"])

    def test_multicol_numpy_array(self):
        """Test vector matching with a multi-column numpy array without formulas."""
        rng = np.random.RandomState(123456789)

        # Generate base simulation
        sim = cate_sim("Sigmoidal", n=200, p=1, balance=0.5, random_state=rng)

        # Generate additional covariates using the same treatment assignments
        X1 = simulate_covars(sim["Ts"], balance=0.7, random_state=rng)
        X2 = simulate_covars(sim["Ts"], balance=0.9, random_state=rng)

        # Create multi-column numpy array
        Xs_array = np.column_stack([sim["Xs"].flatten(), X1.flatten(), X2.flatten()])

        # Verify shape
        assert Xs_array.shape == (200, 3)

        # Fit vector matching without specifying a formula
        vm = VectorMatch()
        retained_ids = vm.fit(sim["Ts"], Xs_array)

        # Verify fitting succeeded
        assert vm.is_fitted
        assert isinstance(retained_ids, list)
        assert len(retained_ids) > 0

        # Verify internal representation converted array to DataFrame
        assert isinstance(vm.cleaned_inputs.Xs_df, pd.DataFrame)
        assert vm.cleaned_inputs.Xs_df.shape[1] == 3

        # Column names should be automatically generated
        assert list(vm.cleaned_inputs.Xs_df.columns) == ["X0", "X1", "X2"]

        # Default formula should include all columns
        assert "X0 + X1 + X2" in vm.cleaned_inputs.formula

    def test_propensity_formula_parameter(self):
        """Test that prop_form_rhs parameter works correctly when passed to fit."""
        rng = np.random.RandomState(123456789)

        # Generate base simulation
        sim = cate_sim("Sigmoidal", n=200, p=1, balance=0.5, random_state=rng)

        # Generate additional covariates using the same treatment assignments
        X1 = simulate_covars(sim["Ts"], balance=0.7, random_state=rng)
        X2 = simulate_covars(sim["Ts"], balance=0.9, random_state=rng)

        # Create DataFrame with multiple columns
        Xs_df = pd.DataFrame(
            {"X0": sim["Xs"].flatten(), "X1": X1.flatten(), "X2": X2.flatten()}
        )

        # Create a formula that uses all three columns
        formula = "X0 + np.log(X1 + 1) + X2"

        vm = VectorMatch()
        vm.fit(sim["Ts"], Xs_df, prop_form_rhs=formula)

        # Check that the formula was stored
        assert vm.prop_form_rhs == formula

        # Verify the formula made it through to the cleaned_inputs
        assert formula in vm.cleaned_inputs.formula

    def test_vm_increases_covariate_overlap(self):
        """Test that VM increases covariate overlap between groups."""
        rng = np.random.RandomState(123456789)
        sim_mod = cate_sim("Sigmoidal", n=200, p=1, balance=0.5, random_state=rng)

        vm = VectorMatch(retain_ratio=0.2)
        retained_ids = vm.fit(sim_mod["Ts"], sim_mod["Xs"])

        Ts_tilde = sim_mod["Ts"][retained_ids]
        Xs_tilde = sim_mod["Xs"][retained_ids]

        # Calculate overlap before and after VM
        ov_before = approx_overlap(
            sim_mod["Xs"][sim_mod["Ts"] == 0], sim_mod["Xs"][sim_mod["Ts"] == 1]
        )

        ov_after = approx_overlap(Xs_tilde[Ts_tilde == 0], Xs_tilde[Ts_tilde == 1])

        # Overlap should increase after VM
        assert ov_before < ov_after

    def test_overlap_improvement_varies_with_balance(self):
        """Test that overlap improvement is greater for less balanced data."""
        rng = np.random.RandomState(123456789)

        # Create two datasets with different balance levels
        sim_low_balance = cate_sim(
            "Sigmoidal", n=200, p=1, balance=0.3, random_state=rng
        )
        sim_high_balance = cate_sim(
            "Sigmoidal", n=200, p=1, balance=0.8, random_state=rng
        )

        # Run vector matching on both datasets
        vm_low = VectorMatch(retain_ratio=0.2)
        vm_high = VectorMatch(retain_ratio=0.2)

        # Process low balance data
        retained_ids_low = vm_low.fit(sim_low_balance["Ts"], sim_low_balance["Xs"])
        Ts_tilde_low = sim_low_balance["Ts"][retained_ids_low]
        Xs_tilde_low = sim_low_balance["Xs"][retained_ids_low]

        # Process high balance data
        retained_ids_high = vm_high.fit(sim_high_balance["Ts"], sim_high_balance["Xs"])
        Ts_tilde_high = sim_high_balance["Ts"][retained_ids_high]
        Xs_tilde_high = sim_high_balance["Xs"][retained_ids_high]

        # Calculate overlap before and after VM for low balance
        low_before = approx_overlap(
            sim_low_balance["Xs"][sim_low_balance["Ts"] == 0],
            sim_low_balance["Xs"][sim_low_balance["Ts"] == 1],
        )

        low_after = approx_overlap(
            Xs_tilde_low[Ts_tilde_low == 0], Xs_tilde_low[Ts_tilde_low == 1]
        )

        # Calculate overlap before and after VM for high balance
        high_before = approx_overlap(
            sim_high_balance["Xs"][sim_high_balance["Ts"] == 0],
            sim_high_balance["Xs"][sim_high_balance["Ts"] == 1],
        )

        high_after = approx_overlap(
            Xs_tilde_high[Ts_tilde_high == 0], Xs_tilde_high[Ts_tilde_high == 1]
        )

        # Calculate improvement in overlap
        low_improvement = low_after - low_before
        high_improvement = high_after - high_before

        # The less balanced data should show greater improvement
        assert low_improvement > high_improvement, (
            f"Expected greater overlap improvement for less balanced data. "
            f"Low balance improvement: {low_improvement}, High balance improvement: {high_improvement}"
        )

        # Both should still show positive improvement
        assert (
            low_improvement > 0
        ), "Low balance data should show positive overlap improvement"
        assert (
            high_improvement > 0
        ), "High balance data should show positive overlap improvement"

    def test_categorical_covariate_design_matrix(self):
        """Test that categorical covariates are properly encoded in the design matrix."""
        # Create a feature matrix with mixed types
        np.random.seed(12345)
        n_samples = 100

        # Generate numeric feature
        numeric_feature = np.random.normal(size=n_samples)

        # Generate categorical feature with 3 levels
        categorical_feature = np.random.choice(["red", "green", "blue"], size=n_samples)

        # Generate binary treatment influenced by both features
        # This creates a simple propensity model
        logits = (
            0.5 * numeric_feature
            + 1.0 * (categorical_feature == "red")
            - 0.5 * (categorical_feature == "green")
        )
        probs = 1 / (1 + np.exp(-logits))
        treatments = np.random.binomial(1, probs)

        # Convert to DataFrame
        mixed_df = pd.DataFrame(
            {"numeric": numeric_feature, "color": categorical_feature}
        )

        # Initialize with these features
        cleaner = _CleanInputsVM(treatments, mixed_df)

        # Check the design matrix shape - should have columns for:
        # intercept + numeric + (3-1) dummy variables for color
        expected_cols = 1 + 1 + (3 - 1)  # intercept + numeric + (categories-1)
        assert cleaner.Xs_design.shape[1] == expected_cols

        # Check column names
        column_names = cleaner.Xs_design.columns.tolist()

        # One column should be the intercept
        assert "Intercept" in column_names

        # One column should be the numeric feature
        assert "numeric" in column_names or any(
            "numeric" in col for col in column_names
        )

        # There should be dummy variable columns for the categorical feature
        # Patsy typically creates column names like "color[T.green]"
        cat_cols = [col for col in column_names if "color" in col]
        assert len(cat_cols) == 2  # Should be 2 dummies for 3 categories

        # Verify the dummy coding is correct
        # For each categorical value, check that the corresponding row has correct dummy values
        for i, color in enumerate(mixed_df["color"]):
            # Skip reference category
            if color == "blue":  # Assuming 'blue' is reference
                continue

            # Find corresponding dummy column
            dummy_col = next((col for col in cat_cols if color in col), None)
            assert dummy_col is not None

            # Verify this row has a 1 in the correct dummy column
            assert cleaner.Xs_design.iloc[i][dummy_col] == 1

            # Verify this row has 0s in other dummy columns
            for other_col in cat_cols:
                if other_col != dummy_col:
                    assert cleaner.Xs_design.iloc[i][other_col] == 0
