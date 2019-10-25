import numpy as np
from numba import njit

from .base import IndependenceTest
from ._utils import _CheckInputs


class Dcorr(IndependenceTest):
    """
    Compute the Dcorr test statistic and p-value.

    Attributes
    ----------
    stat : float
        The computed independence test statistic.
    pvalue : float
        The computed independence test p-value.
    """

    def __init__(self, compute_distance=None):
        IndependenceTest.__init__(self, compute_distance=compute_distance)

    def statistic(self, x, y):
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
        check_input = _CheckInputs(x, y, dim=2,
                                   compute_distance=self.compute_distance)
        x, y = check_input()

        distx = self.compute_distance(x)
        disty = self.compute_distance(y)

        stat = _dcorr(distx, disty)
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
                                   compute_distance=self.compute_distance)
        x, y = check_input()

        return super(Dcorr, self).test(x, y, reps, workers)


@njit
def _center_distmat(distx):
    n = distx.shape[0]

    exp_distx = ((distx.sum(axis=0) / (n-2)).reshape(n, -1)
                + (distx.sum(axis=1) / (n-2)).reshape(n, -1)
                - distx.sum() / ((n-1) * (n-2)))
    cent_distx = distx - exp_distx

    return cent_distx


@njit
def _global_cov(distx, disty):
    return np.sum(distx @ disty)


@njit
def _dcorr(distx, disty):
    cent_distx = _center_distmat(distx)
    cent_disty = _center_distmat(disty)

    covar = _global_cov(cent_distx, cent_disty.T)
    varx = _global_cov(cent_distx, cent_distx.T)
    vary = _global_cov(cent_disty, cent_disty.T)

    if varx <= 0 or vary <= 0:
        stat = 0
    else:
        stat = covar / np.real(np.sqrt(varx * vary))

    return stat
