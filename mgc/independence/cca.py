import numpy as np
from scipy.sparse.linalg import svds

from .base import IndependenceTest
from ._utils import _CheckInputs


class CCA(IndependenceTest):
    r"""
    Class for calculating the CCA test statistic and P-value.

    Attributes
    ----------
    stat : float
        The computed CCA statistic.
    pvalue : float
        The computed CCA p-value.
    """

    def __init__(self):
        IndependenceTest.__init__(self)

    def _statistic(self, x, y):
        r"""
        Calculates the CCA test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples. That is, shapes must be `(n, p)` and `(n, q)` where `n`
            is the number of samples and `p` and `q` are the number of
            dimensions.

        Returns
        -------
        stat : float
            The computed CCA statistic.
        """
        # center each matrix
        centx = x - np.mean(x, axis=0)
        centy = y - np.mean(y, axis=0)

        # calculate covariance and variances for inputs
        covar = centx.T @ centy
        varx = centx.T @ centx
        vary = centy.T @ centy


        # if 1-d, don't calculate the svd
        if varx.shape[1] == 1 or vary.shape[1] == 1 or covar.shape[1] == 1:
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
        r"""
        Calculates the CCA test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples. That is, shapes must be `(n, p)` and `(n, q)` where `n`
            is the number of samples and `p` and `q` are the number of
            dimensions.

        Returns
        -------
        stat : float
            The computed CCA statistic.
        """
        check_input = _CheckInputs(x, y, dim=2, reps=reps)
        x, y = check_input()

        # use default permutation test
        return super(CCA, self).test(x, y, reps, workers)
