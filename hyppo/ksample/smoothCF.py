import numpy as np
from numba import jit
from scipy.stats import chi2
from ._utils import _CheckInputs
from hyppo.ksample.base import KSampleTest, KSampleTestOutput


class SmoothCFTest(KSampleTest):
    r"""
    Smooth Characteristic Function test statistic and p-value

    The Smooth Characteristic Function test is a two-sample test that uses
    differences in the smoothed (analytic) characteristic function of
    two data distributions in order to determine how different
    the two data are :footcite:p:`chwialkowski2015fast`.

    Parameters
    ----------
    num_randfreq: integer
        Used to construct random array with size ''(p, q)'' where 'p' is the number of
        dimensions of the data and 'q' is the random frequency at which the
        test is performed. These are the random test points at which test occurs (see notes).
    random_state: integer
        Set random seed for generation of test points

    Notes
    -----
    The test statistic takes on the following form:

    :math:'nW_n\Sigma_n^{-1}W_n'

    As seen in the above formulation, this test-statistic takes the same form as
    the Hotelling :math: 'T^2' statistic. However, the components are
    defined differently in this case. Given data sets
    X and Y, define the following as :math: 'Z_i', the vector of differences:

    .. math::

        Z_i = (k(X_i, T_1) - k(Y_i, T_1), \ldots,
        k(X_i, T_J) - k(Y_i, T_J)) \in mathbb{R}^J

    The above is the vector of differences between kernels at test points, :math: 'T_j'.
    This same formulation is used in the Mean Embedding Test.
    Moving forward, :math: 'W_n' can be defined:

    :math:'W_n = \frac{1}{n} \sum_{i = 1}^n Z_i

    This leaves :math:'\Sigma_n', the covariance matrix as:

    :math:'\Sigma_n = \frac{1}{n}ZZ^T'

    In the specific case of the Smooth Characteristic function test,
    the vector of differences can be defined as follows:

    .. math::

        Z_i = (f(X_i)\sin(X_iT_1) - f(Y_i)\sin(Y_iT_1),
        f(X_i)\cos(X_iT_1) - f(Y_i)\cos(Y_iT_1),\cdots) \in \mathbb{R}^{2J}

    Once :math:'S_n' is calculated, a threshold :math:'r_{\alpha}' corresponding to the
    :math:'1 - \alpha' quantil of a Chi-squared distribution w/ J degrees of freedom
    is chosen. Null is rejected if :math:'S_n' is larger than this threshold.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, num_randfreq=5, random_state=None):

        if random_state:
            self.random_state = random_state
        else:
            self.random_state = None
        self.num_randfreq = num_randfreq
        KSampleTest.__init__(self)

    def statistic(self, x, y):
        r"""
        Calculates the smooth CF test statistic.

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
            The computed Smooth CF statistic.
        """
        _, p = np.shape(x)
        random_frequencies = _gen_random(p, self.num_randfreq, self.random_state)
        difference = _smooth_difference(random_frequencies, x, y)
        return distance(difference, 2 * self.num_randfreq)

    def test(self, x, y):
        r"""
        Calculates the smooth CF test statistic and p-value.

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
            The computed Smooth CF statistic.
        pvalue : float
            The computed smooth CF p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.ksample import SmoothCFTest
        >>> np.random.seed(1234)
        >>> x = np.random.randn(10, 1)
        >>> y = np.random.randn(10, 1)
        >>> stat, pvalue = SmoothCFTest(random_state=1234).test(x, y)
        >>> '%.2f, %.3f' % (stat, pvalue)
        '4.69, 0.910'
        """
        check_input = _CheckInputs(inputs=[x, y], indep_test=None)
        x, y = check_input()

        stat = self.statistic(x, y)
        pvalue = chi2.sf(stat, 2 * self.num_randfreq)
        self.stat = stat
        self.pvalue = pvalue

        return KSampleTestOutput(stat, pvalue)


def _gen_random(dimension, num_randfeatures, random_state):
    """Generates test points for vector of differences"""
    if random_state:
        np.random.seed(random_state)
    return np.random.randn(dimension, num_randfeatures)


@jit(nopython=True, cache=True)
def _smooth(data):
    """Smooth kernel"""
    norms = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        norms[i] = np.sqrt(np.sum(data[i] ** 2))
    w = norms
    w = np.exp(-(w ** 2) / 2)
    return w.reshape(-1, 1)


@jit(nopython=True, cache=True)
def _smooth_cf(data, w, random_frequencies):
    """Vector of differences for Smooth CF"""
    n, _ = data.shape
    _, d = random_frequencies.shape
    mat = data.dot(random_frequencies)
    arr = np.concatenate((np.sin(mat) * w, np.cos(mat) * w), 1)
    return arr


@jit(nopython=True, cache=True)
def _smooth_difference(random_frequencies, X, Y):
    """Vector of differences for Smooth CF"""
    x_smooth = _smooth(X)
    y_smooth = _smooth(Y)
    return _smooth_cf(X, x_smooth, random_frequencies) - _smooth_cf(
        Y, y_smooth, random_frequencies
    )


def distance(difference, num_randfeatures):
    r"""
    Using the vector of differences as defined above,
    calculates the Smooth Characteristic Function statistic in the form:

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
        The computed smooth CF statistic.
    """
    num_samples, _ = np.shape(difference)
    sigma = np.cov(np.transpose(difference))

    mu = np.mean(difference, 0)

    if num_randfeatures == 1:
        stat = float(num_samples * mu ** 2) / float(sigma)
    else:
        stat = num_samples * mu.dot(np.linalg.solve(sigma, np.transpose(mu)))

    return stat