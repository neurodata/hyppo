import numpy as np
from numba import njit

from .base import TimeSeriesTest
from ._utils import _CheckInputs
from ..independence import Dcorr


class DcorrX(TimeSeriesTest):
    """
    Compute the Dcorr test statistic and p-value.

    Attributes
    ----------
    stat : float
        The computed independence test statistic.
    pvalue : float
        The computed independence test p-value.
    """

    def __init__(self, compute_distance=None, max_lag=0):
        TimeSeriesTest.__init__(self, compute_distance=compute_distance)
        self.max_lag = max_lag

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
        check_input = _CheckInputs(x, y, max_lag=self.max_lag,
                                   compute_distance=self.compute_distance)
        x, y = check_input()

        distx = self.compute_distance(x)
        disty = self.compute_distance(y)

        max_lag = self.max_lag
        dep_lag = []
        dcorr = Dcorr()
        dcorr_stat = dcorr._statistic(x, y)
        dep_lag.append(np.maximum(0, dcorr_stat))

        n = distx.shape[0]
        for j in range(1, max_lag+1):
            slice_distx = distx[j:n, j:n]
            slice_disty = disty[0:(n-j), 0:(n-j)]
            stat = dcorr._statistic(slice_distx, slice_disty)
            dep_lag.append((n-j) * np.maximum(0, stat) / n)

        opt_lag = np.argmax(dep_lag)
        stat = np.sum(dep_lag)
        self.stat = stat
        self.opt_lag = opt_lag

        return stat, opt_lag

    def test(self, x, y, reps=1000, workers=1, random_state=None):
        """
        Calulates the HHG test p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices that have shapes depending on the particular
            independence tests (check desired test class for specifics).
        reps : int, optional
            The number of replications used in permutation, by default 1000.
        random_state : int or np.random.RandomState instance, optional
            If already a RandomState instance, use it.
            If seed is an int, return a new RandomState instance seeded with seed.
            If None, use np.random.RandomState. Default is None.

        Returns
        -------
        pvalue : float
            The computed independence test p-value.
        """
        check_input = _CheckInputs(x,
                                   y,
                                   max_lag=self.max_lag,
                                   compute_distance=self.compute_distance)
        x, y = check_input()

        return super(DcorrX, self).test(x, y, reps, workers, random_state)
