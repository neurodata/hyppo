import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises
import warnings
from scipy.stats import gaussian_kde
import pandas as pd

from ...tools import CATE_SIMULATIONS, cate_sim, simulate_covars
from ..cleaners import CleanInputsPM
from ..propensity_model import GeneralisedPropensityModel


class TestCleanInputsPM:
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

        # Test with CleanInputsPM directly
        cleaner = CleanInputsPM(self.Ts_binary, Xs_1d)
        assert cleaner.Xs_df.shape == (self.n_samples, 1)

    def test_nan_detection_covariates(self):
        """Test that NaN values in Xs are detected and proper error is raised"""
        # Create features with NaN
        Xs_with_nan = self.Xs.copy()
        Xs_with_nan[10, 0] = np.nan

        # Test with CleanInputsPM directly
        with pytest.raises(
            ValueError,
            match="Error checking `Xs'. Error: The input contains nan values",
        ):
            cleaner = CleanInputsPM(self.Ts_binary, Xs_with_nan)

    def test_nan_detection_in_treatments(self):
        """Test that NaN values in Ts are detected and proper error is raised"""
        # Create treatments with NaN
        Ts_with_nan = self.Ts_binary.copy()
        Ts_with_nan[5] = np.nan

        # Test with CleanInputsPM directly
        with pytest.raises(
            ValueError,
            match="Error checking `Ts'. Error: The input contains nan values",
        ):
            cleaner = CleanInputsPM(Ts_with_nan, self.Xs)

    def test_ts_categorical_conversion(self):
        """Test that treatments are properly converted to categorical"""
        # Test binary treatment
        cleaner = CleanInputsPM(self.Ts_binary, self.Xs)
        assert set(cleaner.treatment_maps["value_to_code"].keys()) == {0, 1}
        assert cleaner.K == 2

        # Test multi-category treatment
        cleaner = CleanInputsPM(self.Ts_multi, self.Xs)
        assert set(cleaner.treatment_maps["value_to_code"].keys()) == {0, 1, 2}
        assert cleaner.K == 3

        # Test string categorical treatments
        cleaner = CleanInputsPM(self.Ts_strings, self.Xs)
        assert set(cleaner.treatment_maps["value_to_code"].keys()) == {
            "control",
            "treatment_A",
            "treatment_B",
        }
        assert cleaner.K == 3
        # Check that string treatments are properly encoded as numbers
        assert np.all(np.isin(cleaner.Ts_factor, [0, 1, 2]))

        # Test pandas categorical Series
        cleaner = CleanInputsPM(self.Ts_pandas_cat, self.Xs)
        assert set(cleaner.treatment_maps["value_to_code"].keys()) == {"A", "B", "C"}
        assert cleaner.K == 3

    def test_propensity_formula_generation(self):
        """Test formula generation for propensity model"""
        # Test with default formula
        cleaner = CleanInputsPM(self.Ts_binary, self.Xs_df)
        assert cleaner.formula.startswith("Ts ~")
        assert "Feature_0" in cleaner.formula

        # Test with custom formula
        custom_formula = "Feature_0 + np.log(Feature_1)"
        cleaner = CleanInputsPM(
            self.Ts_binary, self.Xs_df, prop_form_rhs=custom_formula
        )
        assert cleaner.formula == f"Ts ~ {custom_formula}"

    def test_design_matrix_construction(self):
        """Test that the design matrix correctly reflects the formula"""
        # Test with default formula (all features included)
        cleaner = CleanInputsPM(self.Ts_binary, self.Xs_df)

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
        cleaner = CleanInputsPM(
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

        # Test with CleanInputsPM directly
        with pytest.raises(ValueError, match="below the minimum of"):
            cleaner = CleanInputsPM(Ts_small, Xs_small)

    def test_sample_count_mismatch(self):
        """Test detection of sample count mismatch between Ts and Xs"""
        # Create mismatched datasets
        Ts_short = self.Ts_binary[:-5]  # 5 samples fewer than Xs

        # Test with CleanInputsPM directly
        with pytest.raises(ValueError, match="Inconsistent number of samples"):
            cleaner = CleanInputsPM(Ts_short, self.Xs)

    def test_custom_formula_with_numpy_array(self):
        """Test error when providing custom formula with numpy array features"""
        # Test with custom formula but numpy array features
        custom_formula = "X0 + X1 + X2"

        # Should raise error because can't use custom formula with non-DataFrame
        with pytest.raises(TypeError, match="propensity formula"):
            cleaner = CleanInputsPM(
                self.Ts_binary, self.Xs, prop_form_rhs=custom_formula
            )

    def test_treatment_factor_encoding(self):
        """Test that treatments are properly encoded as factors"""
        # Test with string treatments
        cleaner = CleanInputsPM(self.Ts_strings, self.Xs)

        # Verify treatment encoding is consistent
        unique_treatments = np.unique(self.Ts_strings)
        for i, treatment in enumerate(unique_treatments):
            # Find all indices where original treatment is this value
            original_indices = np.where(self.Ts_strings == treatment)[0]
            # Check that all those indices have the same encoded value in Ts_factor
            encoded_values = cleaner.Ts_factor[original_indices]
            assert np.all(encoded_values == encoded_values[0])

        # Test with pandas categorical
        cleaner = CleanInputsPM(self.Ts_pandas_cat, self.Xs)

        # The encoding should preserve the order of categories in the pandas Categorical
        for i, cat in enumerate(self.Ts_pandas_cat.cat.categories):
            # Find indices where original category is this value
            original_indices = np.where(self.Ts_pandas_cat == cat)[0]
            # Check that encoded value matches the category index
            assert np.all(cleaner.Ts_factor[original_indices] == i)

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
        cleaner = CleanInputsPM(treatments, mixed_df)

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

        # Check for dummy variable columns
        cat_cols = [col for col in column_names if "color" in col]
        assert len(cat_cols) == 2  # Should be 2 dummies for 3 categories

        # Verify the dummy coding is correct
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
        gpm = GeneralisedPropensityModel()
        assert hasattr(gpm, "is_fitted")
        assert gpm.is_fitted is False
        assert gpm.model is None
        assert gpm.model_result is None
        assert gpm.pred_probs is None

    def test_vector_matching_with_one_oddball_per_group(self):
        """Test vector matching correctly excludes outliers."""
        nrep = 100
        results = []

        # Set seed for reproducibility
        rng = np.random.RandomState(123456789)
        # Generate 20 unique seeds
        seeds = rng.randint(0, 1000000, size=nrep)

        for i in range(nrep):
            Ts = np.concatenate([np.ones(100), np.ones(100) * 2])
            rngi = np.random.RandomState(seeds[i])
            # Create features with outliers at positions 0 and 199
            X1 = np.concatenate([[-4], rngi.uniform(size=198), [5]])
            X2 = np.concatenate([[-4], rngi.uniform(size=198), [5]])
            X3 = rngi.uniform(size=200)
            Xs = np.column_stack([X1, X2, X3])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gpm = GeneralisedPropensityModel()
                retained_ids = gpm.fit_and_vector_match(Ts, Xs, retain_ratio=0.05)

            # Check if samples 0 and 199 are excluded
            excl_sample_0 = 0 not in retained_ids
            excl_sample_199 = 199 not in retained_ids

            # Check if most other samples are included (>95%)
            included_samples = (
                sum([i in retained_ids for i in range(1, 199)]) / 198 > 0.95
            )

            # Success if both outliers excluded and most normal samples included
            results.append(excl_sample_0 and excl_sample_199 and included_samples)

        # Test passes if this works most of the time (>95%)
        assert np.mean(results) > 0.95

    def test_vector_matching_with_string_treatments(self):
        """Test vector matching correctly excludes outliers with string treatment groups."""
        nrep = 100
        results = []

        rng = np.random.RandomState(123456789)
        seeds = rng.randint(0, 1000000, size=nrep)

        for i in range(nrep):
            # Create string treatments
            Ts = np.array(["control"] * 100 + ["treatment"] * 100)

            rngi = np.random.RandomState(seeds[i])

            # Create features with outliers at positions 0 and 199
            X1 = np.concatenate([[-4], rngi.uniform(size=198), [5]])
            X2 = np.concatenate([[-4], rngi.uniform(size=198), [5]])
            X3 = rngi.uniform(size=200)
            Xs = np.column_stack([X1, X2, X3])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gpm = GeneralisedPropensityModel()
                retained_ids = gpm.fit_and_vector_match(Ts, Xs, retain_ratio=0.05)

            # Check if samples 0 and 199 are excluded
            excl_sample_0 = 0 not in retained_ids
            excl_sample_199 = 199 not in retained_ids

            # Check if most other samples are included (>95%)
            included_samples = (
                sum([idx in retained_ids for idx in range(1, 199)]) / 198 > 0.95
            )

            # Success if both outliers excluded and most normal samples included
            results.append(excl_sample_0 and excl_sample_199 and included_samples)

        # Test passes if this works most of the time (>95%)
        assert np.mean(results) > 0.95

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
            gpm_high = GeneralisedPropensityModel()
            retained_high = gpm_high.fit_and_vector_match(
                sim_high["Ts"], sim_high["Xs"]
            )

            # Moderate balance
            sim_mod = cate_sim("Sigmoidal", n=300, p=3, balance=0.7, random_state=rngi)
            gpm_mod = GeneralisedPropensityModel()
            retained_mod = gpm_mod.fit_and_vector_match(sim_mod["Ts"], sim_mod["Xs"])

            # Low balance (fewer samples retained)
            sim_low = cate_sim("Sigmoidal", n=300, p=3, balance=0.4, random_state=rngi)
            gpm_low = GeneralisedPropensityModel()
            retained_low = gpm_low.fit_and_vector_match(sim_low["Ts"], sim_low["Xs"])

            # Check if the number of retained samples decreases as unbalancedness increases
            lengths = [len(retained_high), len(retained_mod), len(retained_low)]
            ranks = np.argsort(np.argsort(lengths))

            # Test passes if the ranks are as expected (2,1,0)
            results.append(np.array_equal(ranks, [2, 1, 0]))

        assert np.mean(results) > 0.8

    def test_warning_when_samples_retained_is_low(self):
        """Test VM throws warning when samples retained is low."""
        rng = np.random.RandomState(123456789)
        sim_low = cate_sim("Sigmoidal", n=200, p=3, balance=0.2, random_state=rng)

        with pytest.warns(UserWarning):
            gpm = GeneralisedPropensityModel()
            gpm.fit_and_vector_match(sim_low["Ts"], sim_low["Xs"], retain_ratio=0.9)

    def test_error_when_no_samples_retained(self):
        """Test VM throws error when no samples are retained."""
        rng = np.random.RandomState(123456789)
        sim_low = cate_sim(
            "Sigmoidal", n=200, p=3, balance=0, alpha=10, beta=1, random_state=rng
        )

        with pytest.raises(ValueError):
            gpm = GeneralisedPropensityModel()
            gpm.fit_and_vector_match(sim_low["Ts"], sim_low["Xs"], retain_ratio=0)

    def test_is_fitted_flag(self):
        """Test that is_fitted flag is properly set after fitting."""
        rng = np.random.RandomState(123456789)
        sim = cate_sim("Sigmoidal", n=200, p=1, balance=0.5, random_state=rng)

        gpm = GeneralisedPropensityModel()
        assert gpm.is_fitted is False

        # Test fit without matching
        gpm.fit(sim["Ts"], sim["Xs"])
        assert gpm.is_fitted is True

        # Test that attempting to fit again raises an error
        with pytest.raises(ValueError, match="already been fit"):
            gpm.fit(sim["Ts"], sim["Xs"])

    def test_vector_match_requires_fit_first(self):
        """Test that vector_match raises an error if called before fit."""
        gpm = GeneralisedPropensityModel()

        with pytest.raises(ValueError, match="Model must be fitted"):
            gpm.vector_match(retain_ratio=0.05)

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
        gpm = GeneralisedPropensityModel()
        gpm.fit(sim["Ts"], Xs_array)

        # Verify fitting succeeded
        assert gpm.is_fitted

        # Perform vector matching
        retained_ids = gpm.vector_match()

        # Verify matching succeeded
        assert isinstance(retained_ids, list)
        assert len(retained_ids) > 0

        # Verify internal representation converted array to DataFrame
        assert isinstance(gpm.cleaned_inputs.Xs_df, pd.DataFrame)
        assert gpm.cleaned_inputs.Xs_df.shape[1] == 3

        # Column names should be automatically generated
        assert list(gpm.cleaned_inputs.Xs_df.columns) == ["X0", "X1", "X2"]

        # Default formula should include all columns
        assert "X0 + X1 + X2" in gpm.cleaned_inputs.formula

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

        gpm = GeneralisedPropensityModel()
        gpm.fit(sim["Ts"], Xs_df, prop_form_rhs=formula)

        # Check that the formula was stored
        assert gpm.prop_form_rhs == formula

        # Verify the formula made it through to the cleaned_inputs
        assert formula in gpm.cleaned_inputs.formula

    def test_vm_increases_covariate_overlap(self):
        """Test that VM increases covariate overlap between groups."""
        rng = np.random.RandomState(123456789)
        sim_mod = cate_sim("Sigmoidal", n=200, p=1, balance=0.5, random_state=rng)

        gpm = GeneralisedPropensityModel()
        retained_ids = gpm.fit_and_vector_match(
            sim_mod["Ts"], sim_mod["Xs"], retain_ratio=0.2
        )

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
        gpm_low = GeneralisedPropensityModel()
        gpm_high = GeneralisedPropensityModel()

        # Process low balance data
        retained_ids_low = gpm_low.fit_and_vector_match(
            sim_low_balance["Ts"], sim_low_balance["Xs"], retain_ratio=0.2
        )
        Ts_tilde_low = sim_low_balance["Ts"][retained_ids_low]
        Xs_tilde_low = sim_low_balance["Xs"][retained_ids_low]

        # Process high balance data
        retained_ids_high = gpm_high.fit_and_vector_match(
            sim_high_balance["Ts"], sim_high_balance["Xs"], retain_ratio=0.2
        )
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

    def test_vector_matching_with_categorical_covariates(self):
        """Test vector matching with a controlled propensity model including categorical covariates."""
        np.random.seed(123456789)
        n_samples = 1000

        # Generate features with known effect on propensity
        numeric1 = np.random.normal(size=n_samples)
        numeric2 = np.random.normal(size=n_samples)
        categorical = np.random.choice(["A", "B", "C"], size=n_samples)

        # Create a propensity model with known coefficients
        logits = (
            0.5 * numeric1
            - 0.3 * numeric2
            + 1.5 * (categorical == "A")
            + 0.5 * (categorical == "B")
        )
        probs = 1 / (1 + np.exp(-logits))
        treatments = np.random.binomial(1, probs)

        # Create DataFrame with all features
        Xs_df = pd.DataFrame(
            {"numeric1": numeric1, "numeric2": numeric2, "category": categorical}
        )

        # Fit vector matching
        gpm = GeneralisedPropensityModel()
        gpm.fit(treatments, Xs_df)
        retained_ids = gpm.vector_match(retain_ratio=0.7)

        # Verify fitting succeeded
        assert gpm.is_fitted
        assert isinstance(retained_ids, list)
        assert len(retained_ids) > 0

        # Check the propensity model coefficients
        model_params = gpm.model_result.params
        param_names = model_params.index.tolist()

        # Should have intercept and coefficients for category levels
        assert "Intercept" in param_names

        # Check if category levels are present in parameters
        category_params = [p for p in param_names if "category" in p]
        assert len(category_params) == 2  # Should have 2 parameters for 3 categories

        # Compare covariate balance before and after matching
        def categorical_imbalance(treatments, categories):
            imbalance = 0
            for cat in ["A", "B", "C"]:
                prop_t0 = np.mean(categories[treatments == 0] == cat)
                prop_t1 = np.mean(categories[treatments == 1] == cat)
                # Add absolute difference to imbalance measure
                imbalance += abs(prop_t0 - prop_t1)
            return imbalance

        # Calculate imbalance before matching
        before_imbalance = categorical_imbalance(treatments, categorical)

        # Calculate imbalance after matching
        retained_Ts = treatments[retained_ids]
        retained_cats = categorical[retained_ids]
        after_imbalance = categorical_imbalance(retained_Ts, retained_cats)

        # Imbalance should decrease after matching
        assert after_imbalance < before_imbalance, (
            f"Expected categorical imbalance to decrease after matching. "
            f"Before: {before_imbalance}, After: {after_imbalance}"
        )

        # Also check numeric variables balance
        def std_mean_diff(treatments, variable):
            mean_t0 = np.mean(variable[treatments == 0])
            mean_t1 = np.mean(variable[treatments == 1])
            pooled_std = np.sqrt(
                (np.var(variable[treatments == 0]) + np.var(variable[treatments == 1]))
                / 2
            )
            return abs(mean_t0 - mean_t1) / pooled_std if pooled_std > 0 else 0

        # Check balance improvement for numeric1
        before_smd_num1 = std_mean_diff(treatments, numeric1)
        after_smd_num1 = std_mean_diff(retained_Ts, numeric1[retained_ids])

        assert after_smd_num1 < before_smd_num1, (
            f"Expected balance to improve for numeric1. "
            f"Before SMD: {before_smd_num1}, After SMD: {after_smd_num1}"
        )

    def test_fit_from_cleaned(self):
        """Test using fit_from_cleaned method with externally created cleaned inputs."""
        rng = np.random.RandomState(123456789)

        # Generate simulation data
        sim = cate_sim("Sigmoidal", n=200, p=2, balance=0.5, random_state=rng)

        # Create cleaned inputs externally
        cleaned_inputs = CleanInputsPM(sim["Ts"], sim["Xs"])

        # Test using fit_from_cleaned
        gpm = GeneralisedPropensityModel()
        gpm._fit_from_cleaned(cleaned_inputs)

        # Check that model was fitted
        assert gpm.is_fitted
        assert gpm.model is not None
        assert gpm.pred_probs is not None

        # Check that we can perform vector matching after
        retained_ids = gpm.vector_match()
        assert len(retained_ids) > 0

    def test_fit_and_vector_match_retains_parameters(self):
        """Test that fit_and_vector_match passes parameters to both fit and vector_match."""
        rng = np.random.RandomState(123456789)
        sim = cate_sim("Sigmoidal", n=200, p=1, balance=0.5, random_state=rng)

        # Set specific parameters for testing
        test_retain_ratio = 0.15
        test_ddx = True
        test_niter = 50

        gpm = GeneralisedPropensityModel()
        gpm.fit_and_vector_match(
            sim["Ts"],
            sim["Xs"],
            ddx=test_ddx,
            niter=test_niter,
            retain_ratio=test_retain_ratio,
        )

        # Check that parameters were correctly passed and stored
        assert gpm.ddx == test_ddx
        assert gpm.niter == test_niter
        assert gpm.retain_ratio == test_retain_ratio

        # Verify both fitting and matching occurred
        assert gpm.is_fitted
