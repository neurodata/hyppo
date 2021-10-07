from .base import MultivariateTest, MultivariateTestOutput
from ..tools import compute_kern


class dHsic(MultivariateTest):
    def __init__(self, compute_kernel="Gaussian", bandwidth=1, bias=True, **kwargs):
        pass

    def statistic(self, *data_matrices):
        """
        Helper function that calculates the dHsic test statistic.

        Parameters
        ----------
        *data_matrices: Tuple[np.ndarray]
            Input data matrices.

        Returns
        -------
        stat : float
            The computed dHsic statistic.
        """
        pass

    def test(self, *data_matrices, reps=1000, workers=1, auto=True):
        """
        Calculates the dHsic test statistic and p-value.

        Parameters
        ----------
        *data_matrices: Tuple[np.ndarray]
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


def _dhsic(*data_matrices, kernel="Gaussian", bandwidth=1) -> float:
    "Computes dHsic test statistic"
    pass
