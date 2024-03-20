from typing import NamedTuple

from ..independence import CCA
from ._utils import _CheckInputs, compute_stat
from .base import TimeSeriesTest


class CCAXTestOutput(NamedTuple):
    stat: float
    pvalue: float
    dcorrx_dict: dict


class CCAX(TimeSeriesTest):
    r"""
    Cross CCA (CCA-X) test statistic and p-value.

    The p-value returned is calculated using a permutation test.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, max_lag=0, **kwargs):
        TimeSeriesTest.__init__(
            self, max_lag=max_lag, **kwargs
        )

    def statistic(self, x, y):
        r"""
        Helper function that calculates the CCA-X test statistic.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, ``x`` and ``y`` can be distance matrices,
            where the shapes must both be ``(n, n)``.

        Returns
        -------
        stat : float
            The computed CCA-X statistic.
        opt_lag : int
            The computed optimal lag.
        """
        stat, opt_lag = compute_stat(
            x, y, CCA, self.compute_distance, self.max_lag, **self.kwargs
        )
        self.stat = stat
        self.opt_lag = opt_lag

        return stat, opt_lag

    def test(self, x, y, reps=1000, workers=1, random_state=None):
        r"""
        Calculates the DcorrX test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, ``x`` and ``y`` can be distance matrices,
            where the shapes must both be ``(n, n)``.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed DcorrX statistic.
        pvalue : float
            The computed DcorrX p-value.
        dcorrx_dict : dict
            Contains additional useful returns containing the following keys:

                - opt_lag : int
                    The optimal lag that maximizes the strength of the relationship.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.time_series import DcorrX
        >>> np.random.seed(456)
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue, dcorrx_dict = DcorrX().test(x, y, reps = 100)
        >>> '%.1f, %.2f, %d' % (stat, pvalue, dcorrx_dict['opt_lag'])
        '1.0, 0.05, 0'
        """
        check_input = _CheckInputs(
            x,
            y,
            max_lag=self.max_lag,
        )
        x, y, self.max_lag = check_input()

        stat, pvalue, stat_list = super(CCAX(), self).test(
            x, y, reps, workers, random_state
        )
        ccax_dict = {"opt_lag": stat_list[1]}
        return CCAXTestOutput(stat, pvalue, ccax_dict)
