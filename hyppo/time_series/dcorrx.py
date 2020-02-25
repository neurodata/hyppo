import numpy as np
from numba import njit

from .base import TimeSeriesTest
from ._utils import _CheckInputs, compute_stat
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
        check_input = _CheckInputs(
            x, y, max_lag=self.max_lag, compute_distance=self.compute_distance
        )
        x, y = check_input()

        stat, opt_lag = compute_stat(x, y, Dcorr, self.compute_distance, self.max_lag)
        self.stat = stat
        self.opt_lag = opt_lag

        return stat, opt_lag

    def test(self, x, y, reps=1000, workers=1):
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
        check_input = _CheckInputs(
            x, y, max_lag=self.max_lag, compute_distance=self.compute_distance
        )
        x, y = check_input()

        return super(DcorrX, self).test(x, y, reps, workers)
