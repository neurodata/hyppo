import numpy as np
from numba import jit
from scipy.stats import f

from ._utils import _CheckInputs
from .base import KSampleTest


class MANOVA(KSampleTest):
    r"""
    Multivariate analysis of variance (MANOVA) test statistic and p-value.

    MANOVA is the current standard for multivariate `k`-sample testing.
    The test statistic is formulated as below `[1]`_:

    In MANOVA, we are testing if the mean vectors of each of the `k`-samples are the
    same. Define
    :math:`\{ {x_1}_i \stackrel{iid}{\sim} F_{X_1},\ i = 1, ..., n_1 \}`,
    :math:`\{ {x_2}_j \stackrel{iid}{\sim} F_{X_2},\ j = 1, ..., n_2 \}`, ... as `k`
    groups
    of samples deriving from different a multivariate Gaussian distribution with the
    same dimensionality and same covariance matrix.
    That is, the null and alternate hypotheses are,

    .. math::

       H_0 &: \mu_1 = \mu_2 = \cdots = \mu_k, \\
       H_A &: \exists \ j \neq j' \text{ s.t. } \mu_j \neq \mu_{j'}

    Let :math:`\bar{x}_{i \cdot}` refer to the columnwise means of :math:`x_i`; that is,
    :math:`\bar{x}_{i \cdot} = (1/n_i) \sum_{j=1}^{n_i} x_{ij}`. The pooled sample
    covariance of each group, :math:`W`, is

    .. math::

       W = \sum_{i=1}^k \sum_{j=1}^{n_i} (x_{ij} - \bar{x}_{i\cdot}
       (x_{ij} - \bar{x}_{i\cdot})^T

    Next, define :math:`B` as the  sample covariance matrix of the means. If
    :math:`n = \sum_{i=1}^k n_i` and the grand mean is
    :math:`\bar{x}_{\cdot \cdot} = (1/n) \sum_{i=1}^k \sum_{j=1}^{n} x_{ij}`,

    .. math::

       B = \sum_{i=1}^k n_i (\bar{x}_{i \cdot} - \bar{x}_{\cdot \cdot})
       (\bar{x}_{i \cdot} - \bar{x}_{\cdot \cdot})^T

    Some of the most common statistics used when performing MANOVA include the Wilks'
    Lambda, the Lawley-Hotelling trace, Roy's greatest root, and
    Pillai-Bartlett trace (PBT) `[3]`_ `[4]`_ (PBT was chosen to be the best of these
    as it is the most conservative `[5]`_ `[6]`_) and `[7]`_ has shown that there are
    minimal differences in statistical power among these statistics.
    Let :math:`\lambda_1, \lambda_2, \ldots, \lambda_s` refer to the eigenvalues of
    :math:`W^{-1} B`. Here :math:`s = \min(\nu_{B}, p)` is the minimum between the
    degrees of freedom of :math:`B`, :math:`\nu_{B}` and :math:`p`. So, the PBT
    MANOVA test statistic can be written as `[8]`_,

    .. math::

       \mathrm{MANOVA}_{n_1, \ldots, n_k} (x, y) = \sum_{i=1}^s
       \frac{\lambda_i}{1 + \lambda_i} = \mathrm{tr} (B (B + W)^{-1})

    The p-value analytically by using the F statitic. In the case of PBT, given
    :math:`m = (|p - \nu_{B}| - 1) / 2` and :math:`r = (\nu_{W} - p - 1) / 2`, this is
    `[2]`_:

    .. math::

       F_{s(2m + s + 1), s(2r + s + 1)} = \frac{(2r + s + 1)
       \mathrm{MANOVA}_{n_1, n_2} (x, y)}{(2m + s + 1) (s -
       \mathrm{MANOVA}_{n_1, n_2} (x, y))}

    .. _[1]: https://arxiv.org/pdf/1910.08883.pdf
    .. _[2]: https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Multivariate_Analysis_of_Variance-MANOVA.pdf
    .. _[3]: https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/note-on-tests-of-significance-in-multivariate-analysis/068B27CA9134D07C695539F56817871D
    .. _[4]: https://www.jstor.org/stable/2332629?seq=1
    .. _[5]: https://search.proquest.com/openview/3cb813edb2a65685d54f300dd12682a2/1.pdf?pq-origsite=gscholar&cbl=60977
    .. _[6]: https://scholarworks.umass.edu/pare/vol19/iss1/17/
    .. _[7]: https://www.jstor.org/stable/2286719?seq=1
    .. _[8]: http://ibgwww.colorado.edu/~carey/p7291dir/handouts/manova1.pdf
    """

    def __init__(self):
        KSampleTest.__init__(self)

    def statistic(self, *args):
        r"""
        Calulates the MANOVA test statistic.

        Parameters
        ----------
        *args : ndarray
            Variable length input data matrices. All inputs must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)`, ... where `n`, `m`, ... are the number of samples and `p` is
            the number of dimensions.

        Returns
        -------
        stat : float
            The computed MANOVA statistic.
        """
        cmean = tuple(i.mean(axis=0) for i in args)
        gmean = np.vstack(args).mean(axis=0)
        W = _compute_w(args, cmean)
        B = _compute_b(args, cmean, gmean)

        stat = np.trace(B @ np.linalg.pinv(B + W))
        self.stat = stat

        return stat

    def test(self, *args):
        r"""
        Calculates the MANOVA test statistic and p-value.

        Parameters
        ----------
        *args : ndarray
            Variable length input data matrices. All inputs must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)`, ... where `n`, `m`, ... are the number of samples and `p` is
            the number of dimensions.

        Returns
        -------
        stat : float
            The computed MANOVA statistic.
        pvalue : float
            The computed MANOVA p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.ksample import MANOVA
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = MANOVA().test(x, y)
        >>> '%.3f, %.1f' % (stat, pvalue)
        '0.000, 1.0'
        """
        inputs = list(args)
        check_input = _CheckInputs(
            inputs=inputs,
        )
        inputs = check_input()

        N = np.sum([i.shape[0] for i in inputs])
        p = inputs[0].shape[1]
        nu_w = N - len(inputs)

        if nu_w < p:
            raise ValueError("Test cannot be run, degree of freedoms is off")

        stat = self.statistic(*inputs)
        nu_b = len(inputs) - 1
        s = np.min([p, nu_b])
        m = (np.abs(p - nu_b) - 1) / 2
        n = (nu_w - p - 1) / 2
        num = 2 * n + s + 1
        denom = 2 * m + s + 1
        pvalue = f.sf(num / denom * stat / (s - stat), s * denom, s * num)
        self.stat = stat
        self.pvalue = pvalue
        self.null_dist = None

        return stat, pvalue


@jit(nopython=True, cache=True)
def _compute_w(inputs, cmean):  # pragma: no cover
    """Calculate the W matrix"""

    p = list(inputs)[0].shape[1]
    W = np.zeros((p, p))

    for i in range(len(inputs)):
        for j in range(inputs[i].shape[0]):
            W += (inputs[i][j, :] - cmean[i]) @ (inputs[i][j, :] - cmean[i]).T

    return W


@jit(nopython=True, cache=True)
def _compute_b(inputs, cmean, gmean):  # pragma: no cover
    """Calculate the B matrix"""

    p = list(inputs)[0].shape[1]
    B = np.zeros((p, p))

    for i in range(len(inputs)):
        n = inputs[i].shape[0]
        B += n * (cmean[i] - gmean) @ (cmean[i] - gmean).T

    return B
