import numpy as np

from .base import IndependenceTest
from ._utils import _CheckInputs


class RVCorr(IndependenceTest):
    """
    Compute the RV test statistic and p-value.

    Attributes
    ----------
    stat : float
        The computed independence test statistic.
    pvalue : float
        The computed independence test p-value.
    """

    def __init__(self):
        IndependenceTest.__init__(self)

    def statistic(self, x, y):
        """
        Calulates the RV test statistic.

        [Further Description]

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
        check_input = _CheckInputs(x, y, dim=2)
        x, y = check_input()

        centx = x - np.mean(x, axis=0)
        centy = y - np.mean(y, axis=0)

        # calculate covariance and variances for inputs
        covar = centx.T @ centy
        varx = centx.T @ centx
        vary = centy.T @ centy

        covar = np.trace(covar @ covar.T)
        stat = np.divide(covar, np.sqrt(np.trace(varx @ varx)) *
                         np.sqrt(vary @ vary))
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=-1):
        """
        Calulates the RV test p-value.

        [Further Description]

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
        check_input = _CheckInputs(x, y, dim=2, reps=reps)
        x, y = check_input()

        return super(RVCorr, self).test(x, y, reps, workers)
