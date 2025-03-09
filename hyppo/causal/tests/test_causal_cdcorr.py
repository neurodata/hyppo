import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import euclidean_distances
from ..causal_cdcorr import CausalCDcorr


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
