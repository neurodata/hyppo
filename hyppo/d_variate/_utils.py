import numpy as np

from sklearn.utils import check_array
from ..tools import check_reps, contains_nan


class _CheckInputs:
    """Checks inputs for d_variate independence tests"""

    def __init__(self, *args, reps=None):
        self.args = args
        self.reps = reps

    def __call__(self):
        for mat in self.args:
            check_array(mat)
            contains_nan(mat)
            self._check_dim(mat)
            self._convert_float64(mat)
            self._check_min_samples(mat)
            self._check_variance(mat)

        if self.reps:
            check_reps(self.reps)

        return self.args

    def _check_dim(self, mat):
        """Converts input data matrices to proper dimensions"""
        if mat.ndim == 1:
            mat = mat[:, np.newaxis]
        elif mat.ndim != 2:
            raise ValueError(
                "Expected a 2-D array, found shape " "{}".format(mat.shape)
            )

        self._check_nd_indeptest(mat)

        return mat

    def _convert_float64(self, mat):
        """Converts input data matrices to np.float 64 (if not already done)"""
        mat = np.asarray(mat).astype(np.float64)

        return mat

    def _check_nd_indeptest(self, mat):
        """Check if number of samples is the same"""
        samples = []
        n, _ = mat.shape
        samples.append(n)
        if any(sample != samples[0] for sample in samples):
            raise ValueError(
                "Shape mismatch, all input data matrices must have shape "
                "[n, p] and [n, q]."
            )

    def _check_min_samples(self, mat):
        """Check if the number of samples is at least 3"""
        samples = []
        n = mat.shape[0]
        samples.append(n)
        if any(sample <= 2 for sample in samples):
            raise ValueError("Number of samples is too low. Must be at least 3.")

    def _check_variance(self, mat):
        if np.var(mat) == 0:
            raise ValueError("Test cannot be run, one of the inputs has 0 variance")
