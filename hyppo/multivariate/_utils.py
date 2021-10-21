import numpy as np

from sklearn.utils import check_array
from ..tools import check_reps, contains_nan


class _CheckInputs:
    """Checks inputs for multivariate independence tests"""

    def __init__(self, *data_matrices, reps=None):
        self.data_matrices = data_matrices
        self.reps = reps

    def __call__(self):
        for mat in self.data_matrices:
            check_array(mat)
            contains_nan(mat)
        self.data_matrices = self.check_dim()
        self.data_matrices = self.convert_float64()
        self._check_min_samples()
        self._check_variance()

        if self.reps:
            check_reps(self.reps)

        return self.data_matrices

    def check_dim(self):
        """Converts input data matrices to proper dimensions"""
        for mat in self.data_matrices:
            if mat.ndim == 1:
                mat = mat[:, np.newaxis]
            elif mat.ndim != 2:
                raise ValueError(
                    "Expected a 2-D array, found shape " "{}".format(mat.shape)
                )

        self._check_nd_indeptest()

        return self.data_matrices

    def convert_float64(self):
        """Converts input data matrices to np.float 64 (if not already done)"""
        for mat in self.data_matrices:
            mat = np.asarray(mat).astype(np.float64)

        return self.data_matrices

    def _check_nd_indeptest(self):
        """Check if number of samples is the same"""
        samples = []
        for mat in self.data_matrices:
            n, _ = mat.shape
            samples.append(n)
        result = all(sample == samples[0] for sample in samples)
        if result is False:
            raise ValueError(
                "Shape mismatch, all input data matrices must have shape " "[n, p] and [n, q]."
            )

    def _check_min_samples(self):
        """Check if the number of samples is at least 3"""
        samples = []
        for mat in self.data_matrices:
            n = mat.shape[0]
            samples.append(n)
        result = all(sample <= 3 for sample in samples)
        if result is False:
            raise ValueError(
                "Number of samples is too low. Must be at least 3."
            )

    def _check_variance(self):
        for mat in self.data_matrices:
            if np.var(mat) == 0:
                raise ValueError("Test cannot be run, one of the inputs has 0 variance")
