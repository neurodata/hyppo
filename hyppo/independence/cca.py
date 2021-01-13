import numpy as np

from ._utils import _CheckInputs
from .base import IndependenceTest


class CCA(IndependenceTest):
    r"""
    Class for calculating the CCA test statistic and p-value.

    This test can be thought of inferring information from cross-covariance
    matrices [#1CCA]_. It has been thought that virtually all parametric tests
    of significance can be treated as a special case of CCA [#2CCA]_. The
    method was first introduced by Harold Hotelling in 1936 [#3CCA]_.

    See Also
    --------
    RV : RV test statistic and p-value.

    Notes
    -----
    The statistic can be derived as follows [#4CCA]_:

    Let :math:`x` and :math:`y` be `:math:`(n, p)` samples of random variables
    :math:`X` and :math:`Y`. We can center :math:`x` and :math:`y` and then
    calculate the sample covariance matrix :math:`\hat{\Sigma}_{xy} = x^T y`
    and the variance matrices for :math:`x` and :math:`y` are defined
    similarly. Then, the CCA test statistic is found by calculating vectors
    :math:`a \in \mathbb{R}^p` and :math:`b \in \mathbb{R}^q` that maximize

    .. math::

        \mathrm{CCA}_n (x, y) =
            \max_{a \in \mathbb{R}^p, b \in \mathbb{R}^q}
            \frac{a^T \hat{\Sigma}_{xy} b}
            {\sqrt{a^T \hat{\Sigma}_{xx} a}
             \sqrt{b^T \hat{\Sigma}_{yy} b}}

    The p-value returned is calculated using a permutation test using a
    `permutation test <https://hyppo.neurodata.io/reference/tools.html#permutation-test>`_.

    References
    ----------
    .. [#1CCA] Härdle, W. K., & Simar, L. (2015). Canonical correlation
               analysis. In Applied multivariate statistical analysis (pp.
               443-454). Springer, Berlin, Heidelberg.
    .. [#2CCA] Knapp, T. R. (1978). Canonical correlation analysis: A general
               parametric significance-testing system. *Psychological
               Bulletin*, 85(2), 410.
    .. [#3CCA] Hotelling, H. (1992). Relations between two sets of variates.
               In Breakthroughs in statistics (pp. 162-190). Springer, New
               York, NY.
    .. [#4CCA] Hardoon, D. R., Szedmak, S., & Shawe-Taylor, J. (2004).
               Canonical correlation analysis: An overview with application to
               learning methods. Neural computation, 16(12), 2639-2664.
    """

    def __init__(self):
        IndependenceTest.__init__(self)

    def _statistic(self, x, y):
        r"""
        Helper function that calculates the CCA test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples and dimensions. That is, the shapes must be `(n, p)` where
            `n` is the number of samples and `p` is the number of dimensions.

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
        if varx.size == 1 or vary.size == 1 or covar.size == 1:
            covar = np.sum(covar ** 2)
            stat = covar / np.sqrt(np.sum(varx ** 2) * np.sum(vary ** 2))
        else:
            covar = np.sum(np.linalg.svd(covar, 1)[1] ** 2)
            stat = covar / np.sqrt(
                np.sum(np.linalg.svd(varx, 1)[1] ** 2)
                * np.sum(np.linalg.svd(vary, 1)[1] ** 2)
            )
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1):
        r"""
        Calculates the CCA test statistic and p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples and dimensions. That is, the shapes must be `(n, p)` where
            `n` is the number of samples and `p` is the number of dimensions.
        reps : int, optional (default: 1000)
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, optional (default: 1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed CCA statistic.
        pvalue : float
            The computed CCA p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.independence import CCA
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = CCA().test(x, y)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '1.0, 0.00'

        The number of replications can give p-values with higher confidence
        (greater alpha levels).

        >>> import numpy as np
        >>> from hyppo.independence import CCA
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = CCA().test(x, y, reps=10000)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '1.0, 0.00'
        """
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        # use default permutation test
        return super(CCA, self).test(x, y, reps, workers, is_distsim=False)
