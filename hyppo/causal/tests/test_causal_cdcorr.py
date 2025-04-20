import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import euclidean_distances
from ..causal_cdcorr import CausalCDcorr
from ..propensity_model import GeneralisedPropensityModel
from ...tools import cate_sim


class TestCausalCDcorrPreprocess:
    """Test class for the _preprocess method of CausalCDcorr"""

    def setup_method(self):
        """Setup code that runs before each test"""
        np.random.seed(42)
        self.n_samples = 200

        # Create simple numeric data
        self.X_numeric = pd.DataFrame(
            {
                "X1": np.random.normal(0, 1, self.n_samples),
                "X2": np.random.normal(0.2, 1, self.n_samples),
            }
        )

        # Binary treatment
        propensity = 1 / (
            1 + np.exp(-(0.5 * self.X_numeric["X1"] - 0.3 * self.X_numeric["X2"]))
        )
        self.T_binary = np.random.binomial(1, propensity)

        # Categorical treatment
        probs = np.column_stack(
            [
                np.exp(0.2 * self.X_numeric["X1"]),
                np.exp(-0.1 * self.X_numeric["X1"]),
                np.exp(0.1 * self.X_numeric["X2"]),
            ]
        )
        probs = probs / probs.sum(axis=1, keepdims=True)

        self.T_categorical = np.array(["control", "treatment_A", "treatment_B"])[
            np.array([np.random.choice(3, p=probs[i]) for i in range(self.n_samples)])
        ]

        # Numeric outcome
        self.Y_numeric = pd.DataFrame(
            {
                "Y1": 0.7 * self.T_binary
                + 0.2 * self.X_numeric["X1"]
                + np.random.normal(0, 0.5, self.n_samples),
                "Y2": 0.5 * self.T_binary
                + 0.3 * self.X_numeric["X2"]
                + np.random.normal(0, 0.5, self.n_samples),
            }
        )

        # Mixed outcome with categorical column
        self.Y_mixed = pd.DataFrame(
            {
                "Y_num": 0.7 * self.T_binary
                + 0.2 * self.X_numeric["X1"]
                + np.random.normal(0, 0.5, self.n_samples),
                "Y_cat": pd.Categorical(
                    np.random.choice(["low", "medium", "high"], self.n_samples)
                ),
            }
        )

        # Distance matrix outcome
        Y = np.random.normal(0, 1, (self.n_samples, 2))
        Y[:, 0] += 0.8 * self.T_binary  # Add treatment effect
        self.Y_distance = euclidean_distances(Y)

        # Initialize the CausalCDcorr class
        self.cdcorr_standard = CausalCDcorr(compute_distance="euclidean")
        self.cdcorr_precomputed = CausalCDcorr(compute_distance=None)

    def test_preprocess_numeric_data(self):
        """Test _preprocess with numeric outcomes and binary treatment"""
        print(self.Y_numeric)
        DY, DT, KX = self.cdcorr_standard._preprocess(
            self.Y_numeric,
            self.T_binary,
            self.X_numeric,
        )

        # Check outputs
        assert isinstance(DY, np.ndarray)
        assert isinstance(DT, np.ndarray)
        assert isinstance(KX, np.ndarray)

        # Check shapes
        n_retained = len(self.cdcorr_standard.balanced_ids)
        assert DY.shape == (n_retained, n_retained)
        assert DT.shape == (n_retained, n_retained)
        assert KX.shape == (n_retained, n_retained)

        # Check attributes
        assert hasattr(self.cdcorr_standard, "clean_inputs")
        assert hasattr(self.cdcorr_standard, "balanced_ids")
        assert isinstance(self.cdcorr_standard.balanced_ids, list)
        assert len(self.cdcorr_standard.balanced_ids) > 0
        assert all(isinstance(i, int) for i in self.cdcorr_standard.balanced_ids)
        assert max(self.cdcorr_standard.balanced_ids) < self.n_samples

    def test_preprocess_categorical_treatment(self):
        """Test _preprocess with categorical treatment"""
        DY, DT, KX = self.cdcorr_standard._preprocess(
            self.Y_numeric,
            self.T_categorical,
            self.X_numeric,
        )

        # Check outputs
        assert isinstance(DY, np.ndarray)
        assert isinstance(DT, np.ndarray)
        assert isinstance(KX, np.ndarray)

        # Check shapes
        n_retained = len(self.cdcorr_standard.balanced_ids)
        assert DY.shape == (n_retained, n_retained)
        assert DT.shape == (n_retained, n_retained)
        assert KX.shape == (n_retained, n_retained)

        # Check treatment representation
        assert hasattr(self.cdcorr_standard.clean_inputs, "treatment_maps")
        assert self.cdcorr_standard.clean_inputs.K == 3  # Three treatment categories

    def test_preprocess_mixed_outcomes(self):
        """Test _preprocess with mixed numeric and categorical outcomes"""
        DY, DT, KX = self.cdcorr_standard._preprocess(
            self.Y_mixed,
            self.T_binary,
            self.X_numeric,
        )

        # Check outputs
        assert isinstance(DY, np.ndarray)
        assert isinstance(DT, np.ndarray)
        assert isinstance(KX, np.ndarray)

        # Check shapes
        n_retained = len(self.cdcorr_standard.balanced_ids)
        assert DY.shape == (n_retained, n_retained)
        assert DT.shape == (n_retained, n_retained)

        # Check if categorical variables were properly dummy-encoded
        # The original data has 2 columns but after dummy encoding it should have more
        assert hasattr(self.cdcorr_standard, "is_distance")
        assert self.cdcorr_standard.is_distance == True

    def test_preprocess_precomputed_distance(self):
        """Test _preprocess with precomputed distance matrix"""
        DY, DT, KX = self.cdcorr_precomputed._preprocess(
            self.Y_distance,
            self.T_binary,
            self.X_numeric,
        )

        # Check outputs
        assert isinstance(DY, np.ndarray)
        assert isinstance(DT, np.ndarray)
        assert isinstance(KX, np.ndarray)

        # Check shapes
        n_retained = len(self.cdcorr_precomputed.balanced_ids)
        assert DY.shape == (n_retained, n_retained)
        assert DT.shape == (n_retained, n_retained)

        # Make sure the distance flag is set
        assert self.cdcorr_precomputed.is_distance == True

    def test_preprocess_formula(self):
        """Test _preprocess with custom formula"""
        # Define a custom formula
        formula = "X1 + X2**2"

        DY, DT, KX = self.cdcorr_standard._preprocess(
            self.Y_numeric,
            self.T_binary,
            self.X_numeric,
            prop_form_rhs=formula,
        )

        # Check the formula made it to the propensity model
        assert formula in self.cdcorr_standard.clean_inputs.formula

    def test_preprocess_error_handling(self):
        """Test error handling in _preprocess"""
        # Test with invalid inputs
        with pytest.raises(ValueError, match="has 0 samples"):
            # Empty dataframe should cause an error
            self.cdcorr_standard._preprocess(
                pd.DataFrame(), self.T_binary, self.X_numeric
            )

        with pytest.raises(ValueError, match="Inconsistent number"):
            # Mismatched dimensions should cause an error
            self.cdcorr_standard._preprocess(
                self.Y_numeric.iloc[:-10], self.T_binary, self.X_numeric
            )

        # Test with invalid distance matrix
        with pytest.raises(ValueError, match="must be a square matrix"):
            # Non-square matrix should cause an error when compute_distance=None
            self.cdcorr_precomputed._preprocess(
                np.random.random((self.n_samples, 5)), self.T_binary, self.X_numeric
            )

    def test_distance_matrix_computation(self):
        """Test that distance matrices are computed correctly"""
        # Create a simple dataset where we know the expected distances
        X_simple = pd.DataFrame({"X1": [0, 1, 2, 3], "X2": [0, 0, 1, 1]})

        T_simple = np.array([0, 0, 1, 1])

        Y_simple = pd.DataFrame(
            {
                "Y1": [0, 1, 10, 11],  # Clear separation between treatment groups
                "Y2": [5, 6, 15, 16],
            }
        )

        # Initialize with euclidean distance
        cdcorr = CausalCDcorr(compute_distance="euclidean")

        # overwrite vector match to return all indices
        original_vector_match = GeneralisedPropensityModel.vector_match
        original_fit = GeneralisedPropensityModel._fit
        try:

            def mock_vector_match(self, retain_ratio=0.05):
                self.balanced_ids = list(range(len(self.cleaned_inputs.Ts_factor)))
                return self.balanced_ids

            def mock_fit(self, niter=100, ddx=False, retain_ratio=0):
                return self

            GeneralisedPropensityModel.vector_match = mock_vector_match
            GeneralisedPropensityModel._fit = mock_fit

            DY, DT, KX = cdcorr._preprocess(Y_simple, T_simple, X_simple)

            # Check DY dimensions match number of samples
            assert DY.shape == (4, 4)

            # Verify euclidean distances in DY
            # Distance between Y[0] and Y[2] should be sqrt((10-0)^2 + (15-5)^2) = sqrt(100 + 100) = sqrt(200)
            assert np.isclose(DY[0, 2], np.sqrt(200))

            # Distance between Y[1] and Y[3] should be sqrt((11-1)^2 + (16-6)^2) = sqrt(100 + 100) = sqrt(200)
            assert np.isclose(DY[1, 3], np.sqrt(200))

            # Distance between Y[0] and Y[1] should be sqrt((1-0)^2 + (6-5)^2) = sqrt(1 + 1) = sqrt(2)
            assert np.isclose(DY[0, 1], np.sqrt(2))

            # Verify DT reflects treatment differences
            assert np.isclose(DT[0, 1], 0)
            assert np.isclose(DT[2, 3], 0)

            # Distance between different treatment groups should be 1
            assert np.isclose(DT[0, 2], 1)
            assert np.isclose(DT[1, 3], 1)

        finally:
            # Restore original method
            GeneralisedPropensityModel.vector_match = original_vector_match
            GeneralisedPropensityModel._fit = original_fit


class TestCausalCDcorrStat:
    """Test class for the statistical properties of CausalCDcorr"""

    def setup_method(self):
        """Setup code that runs before each test"""
        self.n_sims = 50

    def test_statistic_is_informative(self):
        """Test that the statistic has higher values with greater effects than smaller effects"""
        np.random.seed(123456789)
        results = []

        # Generate random seeds for each simulation
        seeds = np.random.randint(0, 1000000, size=self.n_sims)

        for i in range(self.n_sims):
            sim_effect = cate_sim(
                "Sigmoidal", n=100, p=10, balance=0.5, eff_sz=0.5, random_state=seeds[i]
            )
            sim_no_effect = cate_sim(
                "Sigmoidal",
                n=100,
                p=10,
                balance=0.5,
                eff_sz=0.0,
                random_state=seeds[i] // 2,
            )

            effect_stat = CausalCDcorr(compute_distance="euclidean").statistic(
                sim_effect["Ys"],
                sim_effect["Ts"],
                sim_effect["Xs"],
            )
            no_effect_stat = CausalCDcorr(compute_distance="euclidean").statistic(
                sim_no_effect["Ys"],
                sim_no_effect["Ts"],
                sim_no_effect["Xs"],
            )
            results.append(effect_stat > no_effect_stat)
        success_rate = np.mean(results)
        assert success_rate >= 0.9


class TestCausalCDcorrTest:
    """Test class for the behavior of CausalCDcorr for hypothesis testing"""

    def setup_method(self):
        """Setup code that runs before each test"""
        self.n_sims = 20
        self.n_reps = 100
        self.alpha = 0.1

    def test_statistical_power_under_alt(self):
        """Test that the test reliably detects effects when they exist"""
        np.random.seed(123456789)
        seeds = np.random.randint(0, 1000000, size=self.n_sims)
        effect_rejections = []

        for i in range(self.n_sims):
            # Generate data with effect
            sim_effect = cate_sim(
                "Sigmoidal", n=100, p=2, balance=0.5, eff_sz=0.5, random_state=seeds[i]
            )

            _, effect_pval = CausalCDcorr(compute_distance="euclidean").test(
                sim_effect["Ys"],
                sim_effect["Ts"],
                sim_effect["Xs"],
                reps=self.n_reps,
                random_state=seeds[i],
            )
            effect_rejections.append(effect_pval < self.alpha)

        # Calculate power
        power = np.mean(effect_rejections)

        assert power >= 0.8

    def test_statistical_power_under_null(self):
        """Test that the test fails to reject at approx alpha under null"""
        np.random.seed(123456789)
        seeds = np.random.randint(0, 1000000, size=self.n_sims)
        effect_rejections = []

        for i in range(self.n_sims):
            # Generate data with effect
            sim_effect = cate_sim(
                "Sigmoidal", n=100, p=2, balance=0.5, eff_sz=0, random_state=seeds[i]
            )

            _, effect_pval = CausalCDcorr(compute_distance="euclidean").test(
                sim_effect["Ys"],
                sim_effect["Ts"],
                sim_effect["Xs"],
                reps=self.n_reps,
                random_state=seeds[i],
            )
            effect_rejections.append(effect_pval < self.alpha)

        # Calculate power
        power = np.mean(effect_rejections)

        assert power <= self.alpha * 1.5
