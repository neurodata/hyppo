import numpy as np
from numba import njit

from .._utils import euclidean, check_xy_distmat, chi2_approx
from .base import IndependenceTest
from ._utils import _CheckInputs


class Dcorr(IndependenceTest):
    r"""
    Class for calculating the Dcorr test statistic and p-value.

    Dcorr is a measure of dependence between two paired random matrices of
    not necessarily equal dimensions. The coefficient is 0 if and only if the
    matrices are independent. It is an example of an energy distance.

    Parameters
    ----------
    compute_distance : callable(), optional (default: euclidean)
        A function that computes the distance among the samples within each
        data matrix. Set to `None` if `x` and `y` are already distance
        matrices. To call a custom function, either create the distance matrix
        before-hand or create a function of the form ``compute_distance(x)``
        where `x` is the data matrix for which pairwise distances are
        calculated.
    bias : bool (default: False)
        Whether or not to use the biased or unbiased test statistics.

    See Also
    --------
    Hsic : Hilbert-Schmidt independence criterion test statistic and p-value.
    HHG : Heller Heller Gorfine test statistic and p-value.

    Notes
    -----
    The statistic can be derived as follows:

    Let :math:`x` and :math:`y` be :math:`(n, p)` samples of random variables
    :math:`X` and :math:`Y`. Let :math:`D^x` be the :math:`n \times n`
    distance matrix of :math:`x` and :math:`D^y` be the :math:`n \times n` be
    the distance matrix of :math:`y`. The distance covariance is,

    .. math::

        \mathrm{Dcov}_n (x, y) = \frac{1}{n^2} \mathrm{tr} (D^x H D^y H)

    where :math:`\mathrm{tr} (\cdot)` is the trace operator and :math:`H` is
    defined as :math:`H = I - (1/n) J` where :math:`I` is the identity matrix
    and :math:`J` is a matrix of ones. The normalized version of this
    covariance is Dcorr [#1Dcor]_ and is

    .. math::

        \mathrm{Dcorr}_n (x, y) = \frac{\mathrm{Dcov}_n (x, y)}
                                       {\sqrt{\mathrm{Dcov}_n (x, x)
                                              \mathrm{Dcov}_n (y, y)}}

    This version of distance correlation is defined using the following
    centering process where :math:`\mathbb{1}(\cdot)` is the indicator
    function:

    .. math::

        C^x_{ij} = \left[ D^x_{ij} - \frac{1}{n-2} \sum_{t=1}^n D^x_{it}
            - \frac{1}{n-2} \sum_{s=1}^n D^x_{sj}
            + \frac{1}{(n-1) (n-2)} \sum_{s,t=1}^n D^x_{st} \right]
            \mathbb{1}_{i \neq j}

    and similarly for :math:`C^y`. Then, this unbiased Dcorr is,

    .. math::

        \mathrm{UDcov}_n (x, y) = \frac{1}{n (n-3)} \mathrm{tr} (C^x C^y)

    The normalized version of this covariance [#2Dcor]_ is

    .. math::

        \mathrm{UDcorr}_n (x, y) = \frac{\mathrm{UDcov}_n (x, y)}
                                        {\sqrt{\mathrm{UDcov}_n (x, x)
                                               \mathrm{UDcov}_n (y, y)}}

    References
    ----------
    .. [#1Dcor] Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
                Measuring and testing dependence by correlation of distances.
                *The annals of statistics*, 35(6), 2769-2794.
    .. [#2Dcor] Székely, G. J., & Rizzo, M. L. (2014). Partial distance
                correlation with methods for dissimilarities. *The Annals of
                Statistics*, 42(6), 2382-2412.
    """

    def __init__(self, compute_distance=euclidean, bias=False):
        # set is_distance to true if compute_distance is None
        self.is_distance = False
        if not compute_distance:
            self.is_distance = True
        self.bias = bias

        IndependenceTest.__init__(self, compute_distance=compute_distance)

    def _statistic(self, x, y):
        r"""
        Helper function that calculates the Dcorr test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples. That is, the shapes must be `(n, p)` and `(n, q)` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, `x` and `y` can be distance matrices,
            where the shapes must both be `(n, n)`.

        Returns
        -------
        stat : float
            The computed Dcorr statistic.
        """
        distx = x
        disty = y

        if not self.is_distance:
            distx = self.compute_distance(x)
            disty = self.compute_distance(y)

        stat = _dcorr(distx, disty, self.bias)
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1, auto=True, bias=False):
        r"""
        Calculates the Dcorr test statistic and p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples. That is, the shapes must be `(n, p)` and `(n, q)` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, `x` and `y` can be distance matrices,
            where the shapes must both be `(n, n)`.
        reps : int, optional (default: 1000)
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, optional (default: 1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.
        auto : bool (default: True)
            Automatically uses fast approximation when sample size and size of array
            is greater than 20. If True, and sample size is greater than 20, a fast
            chi2 approximation will be run. Parameters ``reps`` and ``workers`` are
            irrelevant in this case.

        Returns
        -------
        stat : float
            The computed Dcorr statistic.
        pvalue : float
            The computed Dcorr p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.independence import Dcorr
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = Dcorr().test(x, y)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '1.0, 0.00'

        The number of replications can give p-values with higher confidence
        (greater alpha levels).

        >>> import numpy as np
        >>> from hyppo.independence import Dcorr
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = Dcorr().test(x, y, reps=10000)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '1.0, 0.00'

        In addition, the inputs can be distance matrices. Using this is the,
        same as before, except the ``compute_distance`` parameter must be set
        to ``None``.

        >>> import numpy as np
        >>> from hyppo.independence import Dcorr
        >>> x = np.ones((10, 10)) - np.identity(10)
        >>> y = 2 * x
        >>> dcorr = Dcorr(compute_distance=None)
        >>> stat, pvalue = dcorr.test(x, y)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '0.0, 1.00'
        """
        check_input = _CheckInputs(
            x, y, reps=reps, compute_distance=self.compute_distance
        )
        x, y = check_input()

        if self.is_distance:
            check_xy_distmat(x, y)

        if auto and x.shape[0] > 20:
            stat, pvalue = chi2_approx(self._statistic, x, y)
            self.stat = stat
            self.pvalue = pvalue
            self.null_dist = None
        else:
            if not self.is_distance:
                x = self.compute_distance(x, workers=workers)
                y = self.compute_distance(y, workers=workers)
                self.is_distance = True
            stat, pvalue = super(Dcorr, self).test(x, y, reps, workers)

        return stat, pvalue


@njit
def _center_distmat(distx, bias):  # pragma: no cover
    """Centers the distance matrices"""
    n = distx.shape[0]

    # double centered distance matrices
    if bias:
        # use sum instead of mean because of numba restrictions
        exp_distx = (
            np.repeat(distx.sum(axis=0) / n, n).reshape(-1, n).T
            + np.repeat(distx.sum(axis=1) / n, n).reshape(-1, n)
            - (distx.sum() / (n * n))
        )
    else:
        exp_distx = (
            np.repeat((distx.sum(axis=0) / (n - 2)), n).reshape(-1, n).T
            + np.repeat((distx.sum(axis=1) / (n - 2)), n).reshape(-1, n)
            - distx.sum() / ((n - 1) * (n - 2))
        )
    cent_distx = distx - exp_distx
    if not bias:
        np.fill_diagonal(cent_distx, 0)

    return cent_distx


@njit
def _dcorr(distx, disty, bias):  # pragma: no cover
    """Calculate the Dcorr test statistic"""
    # center distance matrices
    cent_distx = _center_distmat(distx, bias)
    cent_disty = _center_distmat(disty, bias)

    # calculate covariances and variances
    covar = np.sum(np.multiply(cent_distx, cent_disty.T))
    varx = np.sum(np.multiply(cent_distx, cent_distx.T))
    vary = np.sum(np.multiply(cent_disty, cent_disty.T))

    # stat is 0 with negative variances (would make denominator undefined)
    if varx <= 0 or vary <= 0:
        stat = 0

    # calculate generalized test statistic
    else:
        stat = covar / np.real(np.sqrt(varx * vary))

    return stat
