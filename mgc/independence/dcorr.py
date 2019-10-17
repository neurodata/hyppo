import numpy as np
from numba import njit

from .base import IndependenceTest
from ._utils import _contains_nan, _CheckInputs


@njit
def _center_distmat(distx):
    n = distx.shape[0]

    exp_distx = (((distx.mean(axis=0) * n) / (n-2)).reshape(n, -1)
                + (distx.mean(axis=1 * n) / (n-2)).reshape(n, -1)
                - np.sum(distx) / ((n-1) * (n-2)))
    cent_distx = distx - exp_distx

    return cent_distx


@njit
def _global_cov(distx, disty):
    return np.sum(distx @ disty)


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

    def __init__(self, compute_distance=None, is_paired=False):
        IndependenceTest.__init__(self, compute_distance=compute_distance)
        self.is_paired = is_paired

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
        check_input = _CheckInputs(x, y, dim=np.max([x.shape[0], y.shape[0]]),
                                   compute_distance=self.compute_distance)
        x, y = check_input(Dcorr.__name__)

        distx = self.compute_distance(x)
        disty = self.compute_distance(y)

        cent_distx = _center_distmat(distx)
        cent_disty = _center_distmat(disty)

        covar = _global_cov(cent_distx, cent_disty.T)
        varx = _global_cov(cent_distx, cent_distx.T)
        vary = _global_cov(cent_disty, cent_disty.T)

        if varx <= 0 or vary <= 0:
            stat = 0
        else:
            if self.is_paired:
                n = cent_distx.shape[0]
                stat = (varx * (n-1)/n + vary * (n-1)/n
                        - 2/n * np.trace(cent_distx @ cent_disty.T))
            else:
                stat = covar / np.real(np.sqrt(varx * vary))

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
        check_input = _CheckInputs(x,
                                   y,
                                   dim=np.max([x.shape[0], y.shape[0]]),
                                   compute_distance=self.compute_distance)
        x, y = check_input(Dcorr.__name__)

        return super(Dcorr, self).test(x, y, reps, workers)
