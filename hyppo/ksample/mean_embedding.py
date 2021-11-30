import numpy as np
from numba import jit
from scipy.stats import chi2
from ._utils import _CheckInputs
from .base import KSampleTestOutput, KSampleTest


class MeanEmbeddingTest(KSampleTest):
    r"""
        Mean Embedding test statistic and p-value

        The Mean Embedding test is a two-sample test that uses
        differences in (analytic) mean embeddings of two data
        distributions in a reproducing kernel Hilbert space in order
        to determine how different the data are.

        Notes
        ____

        The test statistic, like the Smooth CF statistic, takes on the following form:

        :math: 'nW_n\Sigma_n^{-1}W_n'

        As seen in the above formulation, this test-statistic takes the same form as
        the Hotelling :math: 'T^2' statistic. However, the components are
        defined differently in this case. Given data sets
        X and Y, define the following as :math: 'Z_i', the vector of differences:

        :math: 'Z_i = (k(X_i, T_1) - k(Y_i, T_1), \ldots, k(X_i, T_J) - k(Y_i, T_J)) \in mathbb{R}^J'

        The above is the vector of differences between kernels at test points, :math: 'T_j'. The kernel
        maps into the reproducing kernel Hilbert space.
        This same formulation is used in the Mean Embedding Test.
        Moving forward, :math: 'W_n' can be defined:

        :math: 'W_n = \frac{1}{n} \sum_{i = 1}^n Z_i

        This leaves :math: '\Sigma_n', the covariance matrix as:

        :math: '\Sigma_n = \frac{1}{n}ZZ^T'

        Once :math: 'S_n' is calculated, a threshold :math: 'r_{\alpha}' corresponding to the
        :math: '1 - \alpha' quantil of a Chi-squared distribution w/ J degrees of freedom
        is chosen. Null is rejected if :math: 'S_n' is larger than this threshold.

        References
        ----------
        .. footbibliography::
        """
    def __init__(self, scale=1, compute_distance=False, bias=False, number_of_random_frequencies=5, **kwargs):
        self.number_of_frequencies = number_of_random_frequencies
        self.scale = scale
        KSampleTest.__init__(
            self, compute_distance=compute_distance, bias=bias, **kwargs
        )

    def statistic(self, x, y):
        _, dimension = np.shape(x)
        obs = vector_of_differences(dimension, x, y, self.number_of_frequencies, self.scale)
        return _distance(obs, self.number_of_frequencies)

    def test(self, x, y, random_state=None):
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
                """
        check_input = _CheckInputs(
            inputs=[x,y],
            indep_test=None
        )
        x, y = check_input()

        stat = self.statistic(x,y)
        pvalue = chi2.sf(stat, self.number_of_frequencies)
        self.stat = stat
        self.pvalue = pvalue

        return KSampleTestOutput(stat, pvalue)


@jit(nopython=True, cache=True)
def get_estimate(data, point, scale):
    z = data - scale * point
    norms = np.zeros(z.shape[0])
    for i in range(z.shape[0]):
        norms[i] = np.sqrt(np.sum(z[i] ** 2))
    z2 = norms**2
    return np.exp(-z2/2.0)


@jit(nopython=True, cache=True)
def _get_difference(point, x, y, scale):
    return get_estimate(x, point, scale) - get_estimate(y, point, scale)


@jit(nopython=True, cache=True)
def vector_of_differences(dim, x, y, number_of_frequencies, scale):
    """Calculates vector of differences using above helpers"""
    points = np.random.randn(number_of_frequencies, dim)
    ra = np.zeros((number_of_frequencies, x.shape[0]))

    for idx, point in enumerate(points):
        ra[idx] = _get_difference(point, x, y, scale)

    return ra.T


def _distance(difference, num_random_features):
    num_samples, _ = np.shape(difference)
    sigma = np.cov(np.transpose(difference))

    mu = np.mean(difference, 0)

    if num_random_features == 1:
        stat = float(num_samples * mu ** 2) / float(sigma)
    else:
        stat = num_samples * mu.dot(np.linalg.solve(sigma, np.transpose(mu)))

    return stat

