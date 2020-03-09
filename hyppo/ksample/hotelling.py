import numpy as np
from numba import njit
from scipy.stats import f


from .._utils import euclidean, gaussian, perm_test
from .base import KSampleTest
from ._utils import _CheckInputs, k_sample_transform


class Hotelling(KSampleTest):
    r"""
    Class for calculating Hotelling :math:`T^2` test statistic and p-value.
    """

    def __init__(self):
        KSampleTest.__init__(self)

    def _statistic(self, x, y):
        r"""
        Calulates the Hotelling :math:`T^2` test statistic.

        Parameters
        ----------
        x, y : ndarrays
            Input data matrices. `x` and `y` must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)` where `n` and `m` are the number of samples and `p` are
            the number of dimensions. Alternatively, inputs can be distance
            matrices, where the shapes must all be `(n, n)`.
        """
        # ported from Hotteling packge in R
        nx = x.shape[0]
        ny = y.shape[0]

        meanx = np.mean(x, axis=0).reshape(-1, 1)
        meany = np.mean(y, axis=0).reshape(-1, 1)

        covx = np.cov(x, rowvar=False)
        covy = np.cov(y, rowvar=False)

        covs = ((nx - 1) * covx + (ny - 1) * covy) / (nx + ny - 2)
        inv_covs = np.linalg.inv(covs)

        stat = np.sum((meanx - meany).T @ inv_covs @ (meanx - meany) * nx * ny / (nx + ny))

        return stat

    def test(self, x, y, reps=1000, workers=1, auto=True):
        r"""
        Calulates the Hotellings :math:`T^2` test statistic and p-value.

        Parameters
        ----------
        x, y : ndarrays
            Input data matrices. `x` and `y` must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)` where `n` and `m` are the number of samples and `p` are
            the number of dimensions. Alternatively, inputs can be distance
            matrices, where the shapes must all be `(n, n)`.
        reps : int, optional (default: 1000)
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, optional (default: 1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.
        auto : bool (default: True)
            Automatically uses analytical p-value calculation rather than a
            generic permutation test. Parameters ``reps`` and ``workers`` are
            irrelevant in this case.

        Returns
        -------
        stat : float
            The computed Hotellings :math:`T^2` statistic.
        pvalue : float
            The computed Hotellings :math:`T^2` p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.ksample import Hotelling
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = Hotelling().test(x, y)
        >>> '%.3f, %.1f' % (stat, pvalue)
        '-0.136, 1.0'

        The number of replications can give p-values with higher confidence
        (greater alpha levels).

        >>> import numpy as np
        >>> from hyppo.ksample import KSample
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = Hotelling().test(x, y, reps=10000, auto=False)
        >>> '%.3f, %.1f' % (stat, pvalue)
        '0.172, 0.0'
        """
        check_input = _CheckInputs(
            inputs=[x, y],
        )
        x, y = check_input()

        stat = self._statistic(x, y)
        if auto:
            nx, p = x.shape
            ny = y.shape[0]
            m = (nx + ny - p - 1)/(p * (nx + ny - 2))
            pvalue = 1 - f.cdf(m*stat, p, nx+ny-p-1)
        else:
            pvalue = perm_test(self._statistic, x, y, reps=reps, workers=workers)

        return stat, pvalue
