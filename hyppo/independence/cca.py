import numpy as np

from ._utils import _CheckInputs
from .base import IndependenceTest


class CCA(IndependenceTest):
    """
    Canonical Correlation Analysis (CCA) test statistic and p-value.
    """

    def __init__(self):
        super().__init__()

    def statistic(self, x, y):
        """
        Calculates the CCA test statistic.

        Parameters
        ----------
        x, y : ndarray of float
            Input data matrices with shapes (n_samples, n_features_x) and
            (n_samples, n_features_y), respectively. Monodimensional arrays are
            allowed and will be reshaped to (n_samples, 1).

        Returns
        -------
        stat : float
            The largest canonical correlation.
        """
        if x.ndim == 1:
            x = x[:, np.newaxis]  # Convert to 2D if x is 1D
        if y.ndim == 1:
            y = y[:, np.newaxis]  # Convert to 2D if y is 1D

        # Center the data
        centx = x - np.mean(x, axis=0)
        centy = y - np.mean(y, axis=0)

        # Calculate covariance matrices
        cov_xx = centx.T @ centx / (x.shape[0] - 1)
        cov_yy = centy.T @ centy / (y.shape[0] - 1)
        cov_xy = centx.T @ centy / (x.shape[0] - 1)
        cov_yx = cov_xy.T  # Cross-covariance transpose

        # Regularize covariance matrices to prevent numerical instability (small epsilon)
        #cov_xx += epsilon * np.eye(cov_xx.shape[0])
        #cov_yy += epsilon * np.eye(cov_yy.shape[0])

        # Solve the generalized eigenvalue problem
        eigvals, _ = np.linalg.eig(np.linalg.inv(cov_xx) @ cov_xy @ np.linalg.inv(cov_yy) @ cov_yx)

        # Canonical correlations are the square roots of the eigenvalues (real parts only)
        canonical_corr = np.sqrt(np.real(eigvals))  # Only eigenvalues are needed

        # Return the strongest canonical correlation
        stat = np.max(canonical_corr)
        return stat



    def test(self, x, y, reps=1000, workers=1, random_state=None):
        """
        Calculates the CCA test statistic and p-value.
        """
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        # Use default permutation test
        return super(CCA, self).test(
            x, y, reps, workers, is_distsim=False, random_state=random_state
        )
