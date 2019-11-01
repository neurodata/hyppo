import numpy as np
from numba import njit
from sklearn.metrics.pairwise import rbf_kernel

from .base import IndependenceTest
from ._utils import _CheckInputs
from . import Dcorr


class Hsic(IndependenceTest):
    """
    Compute the Dcorr test statistic and p-value.

    Attributes
    ----------
    stat : float
        The computed independence test statistic.
    pvalue : float
        The computed independence test p-value.
    """

    def __init__(self, compute_kernel=None):
        # set compute_kernel rbf kernel distance by default
        if not compute_kernel:
            compute_kernel = rbf_kernel
        self.compute_kernel = compute_kernel

    def _statistic(self, x, y):
        """
        Calulates the Dcorr test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices that have shapes depending on the particular
            independence tests (check desired test class for specifics).

        Returns
        -------
        stat : float
            The computed independence test statistic.
        """
        dcorr = Dcorr(compute_distance=self.compute_kernel)
        stat = dcorr.test(x, y)[0]
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=-1):
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
        check_input = _CheckInputs(x, y, dim=2, reps=reps,
                                   compute_distance=self.compute_kernel)
        x, y = check_input()

        dcorr = Dcorr(compute_distance=self.compute_kernel)
        stat, pvalue = dcorr.test(x, y, reps=reps, workers=workers)
        self.stat = stat
        self.pvalue = pvalue

        return stat, pvalue
