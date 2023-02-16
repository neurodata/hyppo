import numpy as np
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
    num_randfreq: int
        Used to construct random array with size ``(p, q)`` where `p` is the number of
        dimensions of the data and `q` is the random frequency at which the
        test is performed. These are the random test points at which test occurs (see notes).

    Notes
    -----
    The test statistic takes on the following form:

    .. math::

        nW_n\Sigma_n^{-1}W_n

    As seen in the above formulation, this test-statistic takes the same form as
    the Hotelling :math:`T^2` statistic. However, the components are
    defined differently in this case. Given data sets
    X and Y, define the following as :math:`Z_i`, the vector of differences:

    .. math::

        Z_i = (k(X_i, T_1) - k(Y_i, T_1), \ldots,
        k(X_i, T_J) - k(Y_i, T_J)) \in \mathbb{R}^J

    The above is the vector of differences between kernels at test points, :math:`T_j`.
    This same formulation is used in the Mean Embedding Test.
    Moving forward, :math:`W_n` can be defined:

    .. math::

        W_n = \frac{1}{n} \sum_{i = 1}^n Z_i

    This leaves :math:`\Sigma_n`, the covariance matrix as:

    .. math::

        \Sigma_n = \frac{1}{n}ZZ^T

    In the specific case of the Smooth Characteristic function test,
    the vector of differences can be defined as follows:

    .. math::

        Z_i = (f(X_i)\sin(X_iT_1) - f(Y_i)\sin(Y_iT_1),
        f(X_i)\cos(X_iT_1) - f(Y_i)\cos(Y_iT_1),\cdots) \in \mathbb{R}^{2J}

    Once :math:`S_n` is calculated, a threshold :math:`r_{\alpha}` corresponding to the
    :math:`1 - \alpha` quantile of a Chi-squared distribution w/ J degrees of freedom
    is chosen. Null is rejected if :math:`S_n` is larger than this threshold.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, num_randfreq=5):

        self.num_randfreq = num_randfreq
        KSampleTest.__init__(self)

    def statistic(self, x, y, random_state):
        r"""
        Calculates the smooth CF test statistic.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            dimensions. That is, the shapes must be ``(n, p)`` and ``(m, p)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.
        random_state: int
            Set random seed for generation of test points

        Returns
        -------
        stat : float
            The computed Smooth CF statistic.
        """
        _, p = np.shape(x)
        if random_state:
            np.random.seed(random_state)
        random_frequencies = np.random.randn(p, self.num_randfreq)

        x_smooth = np.exp(-np.linalg.norm(x, axis=1) ** 2 / 2).reshape(-1, 1)
        y_smooth = np.exp(-np.linalg.norm(y, axis=1) ** 2 / 2).reshape(-1, 1)

        x_mat = x.dot(random_frequencies)
        y_mat = y.dot(random_frequencies)

        x_smooth_cf = np.concatenate(
            (np.sin(x_mat) * x_smooth, np.cos(x_mat) * x_smooth), 1
        )
        y_smooth_cf = np.concatenate(
            (np.sin(y_mat) * y_smooth, np.cos(y_mat) * y_smooth), 1
        )

        difference = x_smooth_cf - y_smooth_cf
        return smooth_cf_distance(difference)

    def test(self, x, y, random_state=None):
        r"""
        Calculates the smooth CF test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            dimensions. That is, the shapes must be ``(n, p)`` and ``(m, p)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.
        random_state: int
            Set random seed for generation of test points

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
        >>> x = np.random.randn(500, 10)
        >>> y = np.random.randn(500, 10)
        >>> stat, pvalue = SmoothCFTest().test(x, y, random_state=1234)
        >>> '%.2f, %.3f' % (stat, pvalue)
        '4.70, 0.910'
        """
        check_input = _CheckInputs(inputs=[x, y], indep_test=None)
        x, y = check_input()

        stat = self.statistic(x, y, random_state)
        pvalue = chi2.sf(stat, 2 * self.num_randfreq)
        self.stat = stat
        self.pvalue = pvalue

        return KSampleTestOutput(stat, pvalue)


def smooth_cf_distance(difference):
    r"""
    Calculates the Smooth CF test statistic using the vector of differences.

    Parameters
    ----------
    difference : ndarray of float
        The vector of differences which indicates distance between mean embeddings.
    num_randfeatures : int
        The number of test frequencies

    Returns
    -------
    stat : float
        The computed smooth CF statistic.
    """
    num_samples, _ = np.shape(difference)
    sigma = np.cov(np.transpose(difference))

    mu = np.mean(difference, 0)

    stat = num_samples * mu.dot(np.linalg.solve(sigma, np.transpose(mu)))

    return stat
