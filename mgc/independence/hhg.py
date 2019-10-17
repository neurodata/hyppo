import warnings

import numpy as np
from numba import njit

from .base import IndependenceTest
from ._utils import _contains_nan, _CheckInputs


class HHG(IndependenceTest):
    """
    Compute the HHG test statistic and p-value.

    Attributes
    ----------
    stat : float
        The computed independence test statistic.
    pvalue : float
        The computed independence test p-value.
    """

    def __init__(self, compute_distance=None):
        IndependenceTest.__init__(self, compute_distance=compute_distance)

    @staticmethod
    @njit
    def _fast_hhg(distx, disty):
        n = distx.shape[0]
        S = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    a = distx[i, :] <= distx[i, j]
                    b = disty[i, :] <= disty[i, j]

                    t11 = np.sum(a * b) - 2
                    t12 = np.sum(a * (1 - b))
                    t21 = np.sum((1 - a) * b)
                    t22 = np.sum((1 - a) * (1 - b))

                    denom = (t11+t12) * (t21+t22) * (t11+t21) * (t12+t22)
                    if denom > 0:
                        S[i, j] = ((n-2) * (t12*t21 - t11*t22) ** 2) / denom

        stat = np.sum(S)

        return stat

    def statistic(self, x, y):
        """
        Calulates the HHG test statistic.

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
        check_input = _CheckInputs(x, y, dim=np.max([x.shape[0], y.shape[0]]),
                                   compute_distance=self.compute_distance)
        x, y = check_input(HHG.__name__)

        distx = self.compute_distance(x)
        disty = self.compute_distance(y)

        stat = self._fast_hhg(distx, disty)

        self.stat = stat

        return stat

    def p_value(self, x, y, reps=1000, workers=-1):
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
        check_input = _CheckInputs(x,
                                   y,
                                   dim=np.max([x.shape[0], y.shape[0]]),
                                   compute_distance=self.compute_distance)
        x, y = check_input(HHG.__name__)

        return super(HHG, self).p_value(x, y, reps, workers)
