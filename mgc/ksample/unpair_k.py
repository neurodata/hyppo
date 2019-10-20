import numpy as np
from numba import njit

from .base import KSampleTest
from ._utils import _CheckInputs, k_sample_transform


class UnpairKSample(KSampleTest):
    """
    Compute the Dcorr test statistic and p-value.

    Attributes
    ----------
    stat : float
        The computed independence test statistic.
    pvalue : float
        The computed independence test p-value.
    """

    def __init__(self, indep_test, compute_distance=None):
        KSampleTest.__init__(self, indep_test, compute_distance=compute_distance)

    def test(self, indep_test=None, reps=1000, workers=-1, *argv):
        """
        Calulates the HHG test p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices that have shapes depending on the particular
            independence tests (check desired test class for specifics).
        reps : int, optional
            The number of replications used in permutation, by default 1000.

        Returns
        -------
        pvalue : float
            The computed independence test p-value.
        """
        inputs = list(range(*argv))
        check_input = _CheckInputs(dim=np.max([i.shape[0]
                                               for i in inputs]),
                                   indep_test=indep_test,
                                   compute_distance=self.compute_distance
                                   *argv)
        inputs = check_input(UnpairKSample.__name__)

        return super(UnpairKSample, self).test(indep_test, reps, workers, *inputs)
