from .base import MultivariateTest, MultivariateTestOutput
from ..tools import multi_compute_kern, multi_perm_test
from ._utils import _CheckInputs

import numpy as np


class dHsic(MultivariateTest):
    def __init__(self, compute_kernel="gaussian", bias=True, **kwargs):
        self.compute_kernel = compute_kernel

        self.is_kernel = False
        if not compute_kernel:
            self.is_kernel = True
        self.bias = bias

        MultivariateTest.__init__(self, compute_distance=None, **kwargs)

    def statistic(self, *data_matrices):
        """
        Helper function that calculates the dHsic test statistic.

        Parameters
        ----------
        *data_matrices: Tuple[np.ndarray]
            Input data matrices.

        Returns
        -------
        stat : float
            The computed dHsic statistic.
        """
        kerns = multi_compute_kern(*data_matrices)
        n = data_matrices[0].shape[0]
        term1 = np.ones((n, n))
        term2 = 1
        term3 = 2 / n * np.ones((n, 1))
        for j in range(len(kerns)):
            term1 = np.multiply(term1, kerns[j])
            term2 = 1 / (n * n) * term2 * np.sum(kerns[j])
            term3 = 1 / n * np.multiply(term3, np.sum(kerns[j], axis=1))
        stat = 1 / (n * n) * np.sum(term1) + term2 - np.sum(term3)

        return stat

    def test(self, *data_matrices, reps=1000, workers=1, auto=True):
        """
        Calculates the dHsic test statistic and p-value.

        Parameters
        ----------
        *data_matrices: Tuple[np.ndarray]
            Input data matrices.
        reps : int, default=1000
            Number of replications used for permutation test.
        workers : int, default=1
            Number of cores.
        auto : boolean, default=True

        Returns
        -------
        stat : float
            The computed dHsic statistic.
        pvalue : float
            The computed dHsic p-value.
        """
        check_input = _CheckInputs(
            *data_matrices,
            reps=reps,
        )
        data_matrices = check_input()

        data_matrices = multi_compute_kern(*data_matrices, metric=self.compute_kernel, **self.kwargs)
        self.is_kernel = True
        stat, pvalue = multi_perm_test(dHsic().statistic, *data_matrices, reps, workers)
        #stat, pvalue = super(dHsic, self).test(*data_matrices, reps, workers)

        return MultivariateTestOutput(stat, pvalue)

