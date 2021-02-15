import numpy as np
from hyppo.ksample._utils import _CheckInputs
from hyppo.ksample.base import KSampleTest
from scipy.stats import f


class Hotelling(KSampleTest):
    r"""
    Hotelling :math:`T^2` test statistic and p-value.

    Hotelling :math:`T^2` is 2-sample multivariate analysis of variance (MANOVA)
    and generalization of Student's t-test in arbitary dimension `[2]`_.
    The test statistic is formulated as below `[1]`_:

    Consider input samples :math:`u_i \stackrel{iid}{\sim} F_U` for :math:`i \in \{ 1, \ldots, n \}` and :math:`v_i \stackrel{iid}{\sim} F_V` for :math:`i \in \{ 1, \ldots, m \}`. Let :math:`\bar{u}` refer to the columnwise means of :math:`u`; that is, :math:`$\bar{u} = (1/n) \sum_{i=1}^{n} u_i` and let :math:`\bar{v}` be the same for :math:`v`. Calculate sample covariance matrices :math:`\hat{\Sigma}_{uv} = u^T v` and sample variance matrices :math:`\hat{\Sigma}_{uu} = u^T u` and :math:`\hat{\Sigma}_{vv} = v^T v`. Denote pooled covariance matrix :math:`\hat{\Sigma}` as

    .. math::

       \hat{\Sigma} = \frac{(n -  1) \hat{\Sigma}_{uu} + (m - 1) \hat{\Sigma}_{vv} }
       {n + m - 2}

    Then,

    .. math::

       \text{\Hotelling}_{n, m} (u, v) = \frac{n m}{n + m}
       (\bar{u} - \bar{v})^T \hat{\Sigma}^{-1} (\bar{u} - \bar{v})

    Since it is a multivariate generalization of Student's t-tests, it suffers from
    some of the same assumptions as Student's t-tests. That is, the validity of MANOVA
    depends on the assumption that random variables are normally distributed within
    each group and each with the same covariance matrix. Distributions of input data
    are generally not known and cannot always be reasonably modeled as Gaussian `[3]`_
    `[4]`_ and having the same covariance across groups is also generally not true of
    real data.

    .. _[1]: https://arxiv.org/pdf/1910.08883.pdf
    .. _[2]: https://projecteuclid.org/euclid.aoms/1177732979
    .. _[3]: https://psycnet.apa.org/record/1989-14214-001
    .. _[4]: https://projecteuclid.org/euclid.aos/1176343997
    """

    def __init__(self):
        KSampleTest.__init__(self)

    def statistic(self, x, y):
        r"""
        Calulates the Hotelling :math:`T^2` test statistic.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            dimensions. That is, the shapes must be ``(n, p)`` and ``(m, p)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.

        Returns
        -------
        stat : float
            The computed Hotelling :math:`T^2` statistic.
        """
        # ported from Hotteling packge in R
        nx, p = x.shape
        ny = y.shape[0]

        meanx = np.mean(x, axis=0)
        meany = np.mean(y, axis=0)

        covx = np.cov(x, rowvar=False)
        covy = np.cov(y, rowvar=False)

        covs = ((nx - 1) * covx + (ny - 1) * covy) / (nx + ny - 2)
        m = (nx + ny - p - 1) / (p * (nx + ny - 2))
        if p > 1:
            inv_covs = np.linalg.pinv(covs)
            stat = (
                m * (meanx - meany).T @ inv_covs @ (meanx - meany) * nx * ny / (nx + ny)
            )
        else:
            inv_covs = 1 / p
            stat = m * (meanx - meany) ** 2 * inv_covs * nx * ny / (nx + ny)

        self.stat = stat

        return stat

    def test(self, x, y):
        r"""
        Calculates the Hotelling :math:`T^2` test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            dimensions. That is, the shapes must be ``(n, p)`` and ``(m, p)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.

        Returns
        -------
        stat : float
            The computed Hotelling :math:`T^2` statistic.
        pvalue : float
            The computed Hotelling :math:`T^2` p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.ksample import Hotelling
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = Hotelling().test(x, y)
        >>> '%.3f, %.1f' % (stat, pvalue)
        '0.000, 1.0'
        """
        check_input = _CheckInputs(
            inputs=[x, y],
        )
        x, y = check_input()

        stat = self.statistic(x, y)
        nx, p = x.shape
        ny = y.shape[0]
        pvalue = f.sf(stat, p, nx + ny - p - 1)
        self.stat = stat
        self.pvalue = pvalue
        self.null_dist = None

        return stat, pvalue
