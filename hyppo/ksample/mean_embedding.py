import numpy as np
from numba import jit
from scipy.stats import chi2
from ._utils import _CheckInputs
from .base import KSampleTestOutput, KSampleTest


class MeanEmbeddingTest(KSampleTest):
    r"""
    Mean Embedding test statistic and p-value.

    The Mean Embedding test is a two-sample test that uses
    differences in (analytic) mean embeddings of two data
    distributions in a reproducing kernel Hilbert space.
    :footcite:p:`chwialkowski2015fast`.

    Parameters
    ----------
    num_randfreq: integer
        Used to construct random array with size ''(p, q)'' where 'p' is the number of
        dimensions of the data and 'q' is the random frequency at which the
        test is performed. These are the random test points at which test occurs (see notes).

    Notes
    -----
    The test statistic, like the Smooth CF statistic, takes on the following form:

    :math:'nW_n\Sigma_n^{-1}W_n'

    As seen in the above formulation, this test-statistic takes the same form as
    the Hotelling :math:'T^2' statistic found in :class:`hyppo.ksample.Hotelling`.
    However, the components are defined differently in this case. Given data sets
    X and Y, define the following as :math: 'Z_i', the vector of differences:

    .. math::

        Z_i = (k(X_i, T_1) - k(Y_i, T_1), \ldots,
        k(X_i, T_J) - k(Y_i, T_J)) \in mathbb{R}^J

    The above is the vector of differences between kernels at test points,
    :math: 'T_j'. The kernel maps into the reproducing kernel Hilbert space.
    This same formulation is used in the Mean Embedding Test.
    Moving forward, :math: 'W_n' can be defined:

    :math:'W_n = \frac{1}{n} \sum_{i = 1}^n Z_i

    This leaves :math: '\Sigma_n', the covariance matrix as:

    :math:'\Sigma_n = \frac{1}{n}ZZ^T'

    Once :math:'S_n' is calculated, a threshold :math:'r_{\alpha}' corresponding to the
    :math:'1 - \alpha' quantile of a Chi-squared distribution w/ J degrees of freedom
    is chosen. Null is rejected if :math: 'S_n' is larger than this threshold.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, num_randfreq=5):
        self.num_randfreq = num_randfreq
        KSampleTest.__init__(self)

    def statistic(self, x, y):
        r"""
        Calculates the mean embedding test statistic.

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
            The computed mean embedding statistic.
        """
        _, p = np.shape(x)
        obs = _vector_of_differences(p, x, y, self.num_randfreq)
        return distance(obs, self.num_randfreq)

    def test(self, x, y):
        r"""
        Calculates the mean embedding test statistic and p-value.

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
            The computed mean embedding statistic.
        pvalue : float
            The computed mean embedding p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.ksample import SmoothCFTest
        >>> np.random.seed(120)
        >>> x = np.random.randn(10, 1)
        >>> y = np.random.randn(10, 1)
        >>> stat, pvalue = SmoothCFTest().test(x, y)
        >>> '%.3f, %.3f' % (stat, pvalue)
        '3.852, 0.571'
        """
        check_input = _CheckInputs(inputs=[x, y], indep_test=None)
        x, y = check_input()

        stat = self.statistic(x, y)
        pvalue = chi2.sf(stat, self.num_randfreq)
        self.stat = stat
        self.pvalue = pvalue

        return KSampleTestOutput(stat, pvalue)


@jit(nopython=True, cache=True)
def _get_estimate(data, point):
    z = data - point
    norms = np.zeros(z.shape[0])
    for i in range(z.shape[0]):
        norms[i] = np.sqrt(np.sum(z[i] ** 2))
    z2 = norms ** 2
    return np.exp(-z2 / 2.0)


@jit(nopython=True, cache=True)
def _get_difference(point, x, y):
    return _get_estimate(x, point) - _get_estimate(y, point)


@jit(nopython=True, cache=True)
def _vector_of_differences(dim, x, y, num_randfreq):
    """Calculates vector of differences using above helpers"""
    points = np.random.randn(num_randfreq, dim)
    ra = np.zeros((num_randfreq, x.shape[0]))

    for idx, point in enumerate(points):
        ra[idx] = _get_difference(point, x, y)

    return ra.T


def distance(difference, num_randfeatures):
    r"""
    Using the vector of differences as defined above,
    calculates the Mean Embedding statistic in the form:

    :math:'nW_n\Sigma_n^{-1}W_n'

    Where :math:'W_n' is the vector of differences.

    Parameters
    ----------
    difference : ndarray
        The vector of differences which indicates distance between mean embeddings.
    num_randfeatures : integer
        The number of test frequencies
    Returns
    -------
    stat : float
        The computed mean embedding statistic.
    """
    num_samples, _ = np.shape(difference)
    sigma = np.cov(np.transpose(difference))

    mu = np.mean(difference, 0)

    if num_randfeatures == 1:
        stat = float(num_samples * mu ** 2) / float(sigma)
    else:
        stat = num_samples * mu.dot(np.linalg.solve(sigma, np.transpose(mu)))

    return stat