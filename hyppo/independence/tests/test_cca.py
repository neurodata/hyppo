import unittest
import numpy as np
from hyppo.independence import CCA
from sklearn.cross_decomposition import CCA as SklearnCCA


class TestCCA(unittest.TestCase):
    def setUp(self):
        # Common test setup: Create sample data
        self.cca = CCA()
        self.sklearn_cca = SklearnCCA(n_components=1)  # Use only the first canonical correlation
        self.x = np.random.rand(100, 3)
        self.y = np.random.rand(100, 3)

    def test_statistic_correctness(self):
        # Fit sklearn CCA and calculate the first canonical correlation
        self.sklearn_cca.fit(self.x, self.y)
        x_c, y_c = self.sklearn_cca.transform(self.x, self.y)
        expected_stat = np.corrcoef(x_c.T, y_c.T)[0, 1]  # First canonical correlation

        # Compute the statistic using the custom implementation
        stat = self.cca.statistic(self.x, self.y)

        # Compare the results
        self.assertAlmostEqual(stat, expected_stat, places=5, msg="Statistic value is incorrect")

    def test_statistic_with_noise(self):
        # Add noise to the data and compute the statistic
        noisy_y = self.y + np.random.normal(0, 0.1, self.y.shape)
        stat = self.cca.statistic(self.x, noisy_y)
        self.assertGreater(stat, 0, "Statistic should be greater than zero for correlated inputs")

    def test_statistic_for_unrelated_data(self):
        # Test with unrelated data
        unrelated_y = np.random.rand(*self.y.shape)
        stat = self.cca.statistic(self.x, unrelated_y)
        self.assertLess(stat, 0.5, "Statistic should be small for uncorrelated inputs")


if __name__ == "__main__":
    unittest.main()
