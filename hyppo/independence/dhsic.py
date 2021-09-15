
import numpy as np

from .base import IndependenceTest, IndependenceTestOutput


class dHsic(IndependenceTest):
    def __init__(self, compute_kernel="Gaussian", bandwidth=1, bias=True, **kwargs):
        pass

    def statistic(self, x, y):
        """
        Helper function that calculates the dHsic test statistic.

        Parameters
        ----------
        x,y: np.ndarray
            Input data matrices.

        Returns
        -------
        stat : float
            The computed dHsic statistic.
        """
        pass

    def test(self, x, y, reps=1000, workers=1, auto=True):
        """
        Calculates the dHsic test statistic and p-value.

        Parameters
        ----------
        x,y : np.ndarray
            Input data matrices.
        reps : int, default=1000
            Number of replications used for permutation test.
        workers : int, default=1
            Number of cores.
        auto : boolean, default=True

        Returns
        -------
        stat : float
            The computed dHsic statistic.
        pvalue : float
            The computed dHsic p-value.
        """
        pass


def _compute_gram(x, kernel="Gaussian", bandwidth=1) -> np.ndarray:
    "Computes gram matrix using kernel given a data matrix x"
    pass


def _dhsic(x, k) -> float:
    "Computes dHsic test statistic given a data matrix x and a gram matrix k"
    pass






