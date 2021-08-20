import numpy as np

from ._utils import _CheckInputs
from .base import IndependenceTest


class CCA(IndependenceTest):
    r"""
    Cannonical Correlation Analysis (CCA) test statistic and p-value.

    This test can be thought of inferring information from cross-covariance
    matrices `[1]`_. It has been thought that virtually all parametric tests
    of significance can be treated as a special case of CCA `[2]`_. The
    method was first introduced by Harold Hotelling in 1936 `[3]`_.

    The statistic can be derived as follows `[4]`_:

    Let :math:`x` and :math:`y` be :math:`(n, p)` samples of random variables
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

    The p-value returned is calculated using a permutation test using
    :meth:`hyppo.tools.perm_test`.

    .. _[1]: https://link.springer.com/book/10.1007/978-3-662-45171-7
    .. _[2]: https://psycnet.apa.org/record/1979-00149-001
    .. _[3]: https://link.springer.com/chapter/10.1007/978-1-4612-4380-9_14
    .. _[4]: https://ieeexplore.ieee.org/document/6788402
    """

    def __init__(self):
        IndependenceTest.__init__(self)

    def statistic(self, x, y):
        r"""
        Helper function that calculates the CCA test statistic.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples and dimensions. That is, the shapes must be ``(n, p)`` where
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
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples and dimensions. That is, the shapes must be ``(n, p)`` where
            `n` is the number of samples and `p` is the number of dimensions.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.

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
        """
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        # use default permutation test
        return super(CCA, self).test(x, y, reps, workers, is_distsim=False)
