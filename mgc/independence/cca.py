import numpy as np
from scipy.sparse.linalg import svds

from .base import IndependenceTest
from ._utils import _CheckInputs


class CannCorr(IndependenceTest):
    """
    Compute the CCA test statistic and p-value.

    Attributes
    ----------
    stat : float
        The computed independence test statistic.
    pvalue : float
        The computed independence test p-value.
    """

    def __init__(self):
        IndependenceTest.__init__(self)

    def _statistic(self, x, y):
        """
        Calulates the CCA test statistic.

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
        centx = x - np.mean(x, axis=0)
        centy = y - np.mean(y, axis=0)

        # calculate covariance and variances for inputs
        covar = centx.T @ centy
        varx = centx.T @ centx
        vary = centy.T @ centy

        if varx.size == 1 or vary.size == 1 or covar.size == 1:
            covar = np.sum(covar ** 2)
            stat = np.divide(covar, np.sqrt(np.sum(varx ** 2) *
                                            np.sum(vary ** 2)))
        else:
            covar = np.sum(svds(covar, 1)[1] ** 2)
            stat = np.divide(covar, np.sqrt(np.sum(svds(varx, 1)[1] ** 2)
                                            * np.sum(svds(vary, 1)[1] ** 2)))
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=-1):
        """
        Calulates the CCA test p-value.

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

        return super(CannCorr, self).test(x, y, reps, workers)
