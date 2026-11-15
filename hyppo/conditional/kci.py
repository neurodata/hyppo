import warnings

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gamma
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from ._utils import _CheckInputs
from .base import ConditionalIndependenceTest, ConditionalIndependenceTestOutput


class KCI(ConditionalIndependenceTest):
    r"""
    Kernel Conditional Independence Test Statistic and P-Value.

    This is a conditional indpendence test utilizing a radial basis
    function to calculate the kernels of two datasets. The trace
    of the normalized matrix product is then calculated to extract the test
    statistic. A Gaussian distribution is then utilized to calculate
    the p-value given the statistic and approximate mean and variance
    of the trace values of the independent kernel matrices.
    This test is consistent against similar tests.

    Notes
    -----
    The statistic is computed as follows :footcite:p:`10.5555/3020548.3020641`:

    Let :math:`x` be a combined sample of :math:`(n, p)` sample
    of random variables :math:`X` and let :math:`y` be a :math:`(n, 1)`
    labels of sample classes :math:`Y`. We can then generate
    :math:`K^x` and :math:`K^y` kernel matrices for each of the respective
    samples. Normalizing, multiplying, and taking the trace of these
    kernel matrices gives the resulting test statistic.
    The p-value and null distribution for the corrected statistic are calculated a
    gamma distribution approximation.

    When a conditioning variable ``z`` is given, the test follows
    :footcite:p:`10.5555/3020548.3020641`: the part of the kernels explained by
    :math:`Z` is removed with the kernel ridge residual operator
    :math:`R_Z = \epsilon (\tilde K_Z + \epsilon I)^{-1}`. If ``z`` is omitted,
    the test is unconditional and a warning is raised.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, **kwargs):

        ConditionalIndependenceTest.__init__(self, **kwargs)

    def compute_kern(self, x, y):

        T = len(y)

        x = np.array(x)
        y = np.array(y)
        x = x - np.mean(x)
        x = x / np.std(x)
        y = y - np.mean(y)
        y = y / np.std(y)

        if T < 200:
            width = 0.8
        elif T < 1200:
            width = 0.5
        else:
            width = 0.3

        theta = 1 / (width**2)

        Kx = 1.0 * RBF(theta).__call__(x, x)
        Ky = 1.0 * RBF(theta).__call__(y, y)

        return Kx, Ky

    @staticmethod
    def _residual_kernels(x, y, z, epsilon=1e-3):
        # constants follow the authors' reference implementation
        x, y, z = _CheckInputs(x, y, z, ignore_z_var=True)()

        def standardize(a):
            s = a.std(axis=0, ddof=1)
            return (a - a.mean(axis=0)) / np.where(s == 0, 1, s)

        x, y, z = standardize(x), standardize(y), standardize(z)
        T = len(y)

        if T < 200:
            width = 1.2
        elif T < 1200:
            width = 0.7
        else:
            width = 0.4

        def gram(a):
            w = width * np.sqrt(z.shape[1])
            return np.exp(-cdist(a, a, "sqeuclidean") / (2 * w**2))

        H = np.eye(T) - np.ones((T, T)) / T
        Kx = H @ gram(np.hstack((x, z / 2))) @ H
        Ky = H @ gram(y) @ H
        Kz = H @ gram(z) @ H

        Rz = epsilon * np.linalg.inv(Kz + epsilon * np.eye(T))
        return Rz @ Kx @ Rz, Rz @ Ky @ Rz

    def _conditional_test(self, x, y, z):
        T = len(y)
        Kx, Ky = self._residual_kernels(x, y, z)
        stat = np.trace(Kx @ Ky)

        # gamma approximation from the eigenvalues (Zhang et al. 2011, Theorem 4)
        wx, vx = np.linalg.eigh((Kx + Kx.T) / 2)
        wy, vy = np.linalg.eigh((Ky + Ky.T) / 2)
        keep_x = wx > wx.max() * 1e-5
        keep_y = wy > wy.max() * 1e-5
        phi_x = vx[:, keep_x] * np.sqrt(wx[keep_x])
        phi_y = vy[:, keep_y] * np.sqrt(wy[keep_y])
        uu = (phi_x[:, :, None] * phi_y[:, None, :]).reshape(T, -1)
        gram = uu @ uu.T
        mean_appr = np.trace(gram)
        var_appr = 2 * np.sum(gram**2)
        k_appr = mean_appr**2 / var_appr
        theta_appr = var_appr / mean_appr
        pvalue = 1 - gamma.cdf(stat, k_appr, scale=theta_appr)

        self.stat = stat
        self.pvalue = pvalue
        return ConditionalIndependenceTestOutput(stat, pvalue)

    @staticmethod
    def _warn_no_z():
        warnings.warn(
            "No conditioning variable z was given, so KCI runs the unconditional "
            "test of x and y. Pass z to test conditional independence.",
            UserWarning,
            stacklevel=3,
        )

    def statistic(self, x, y, z=None):

        if z is not None:
            Kx, Ky = self._residual_kernels(x, y, z)
            return np.trace(Kx @ Ky)

        self._warn_no_z()
        return self._unconditional_statistic(x, y)

    def _unconditional_statistic(self, x, y):

        T = len(y)

        H = np.eye(T) - np.ones((T, T)) / T

        Kx, Ky = self.compute_kern(x, y)

        Kx = (H @ Kx) @ H
        Ky = (H @ Ky) @ H

        stat = np.trace(Kx @ Ky)

        return stat

    def test(self, x, y, z=None):
        r"""
        Calculates the Kernel Conditional Independence test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            columns. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the dimension of samples and `p` is the number of
            dimensions.
        z : ndarray of float, optional
            Conditioning variable of shape ``(n, q)``. If ``None`` (the
            default), the unconditional test is run and a ``UserWarning`` is
            raised.

        Returns
        -------
        stat : float
            The computed Kernel Conditional Independence statistic.
        pvalue : float
            The computed Kernel Conditional Independence p-value.

        Example
        --------
        >>> from hyppo.conditional import KCI
        >>> from hyppo.tools.indep_sim import linear
        >>> np.random.seed(123456789)
        >>> x, y = linear(100, 1)
        >>> stat, pvalue = KCI().test(x, y)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '544.7, 0.00'

        With a conditioning variable ``z``:

        >>> rng = np.random.RandomState(0)
        >>> z = rng.normal(size=(100, 1))
        >>> x = z + 0.5 * rng.normal(size=(100, 1))
        >>> y = z**2 + 0.5 * rng.normal(size=(100, 1))
        >>> stat, pvalue = KCI().test(x, y, z)
        >>> '%.4f, %.4f' % (stat, pvalue)
        '0.9945, 0.1761'
        """

        if z is not None:
            return self._conditional_test(x, y, z)

        self._warn_no_z()

        T = len(y)

        Kx, Ky = self.compute_kern(x, y)
        stat = self._unconditional_statistic(x, y)

        mean_appr = (np.trace(Kx) * np.trace(Ky)) / T
        var_appr = (
            2 * np.trace(Kx @ Kx) * np.trace(Ky @ Ky) / T**2
        )
        k_appr = mean_appr**2 / var_appr
        theta_appr = var_appr / mean_appr
        pvalue = 1 - np.mean(gamma.cdf(stat, k_appr, theta_appr))

        self.stat = stat

        return ConditionalIndependenceTestOutput(stat, pvalue)
