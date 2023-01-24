import numpy as np
from sklearn.utils import check_random_state

from .indep_sim import _CheckInputs


def independent_normal(n, random_state=None):
    r"""
    Conditionally independent normal distributions
    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z`` have shapes
        ``(n, 3)`` where `n` is the number of samples and `3` is the
        number of dimensions.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=3)
    check_in()

    rng = check_random_state(random_state)

    mean = np.array([0, 0, 0])
    cov = np.array([[1, 0.36, 0.6], [0.36, 1, 0.6], [0.6, 0.6, 1]])

    x, y, z = rng.multivariate_normal(mean, cov, size=n).T

    return x, y, z


def independent_lognormal(n, random_state=None):
    r"""
    Conditionally independent normal distributions
    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z`` have shapes
        ``(n, 3)`` where `n` is the number of samples and `3` is the
        number of dimensions.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=3)
    check_in()

    x, y, z = np.exp(independent_normal(n, random_state))

    return x, y, z


def independent_normal_nonlinear(n, random_state=None):
    r"""
    Conditionally independent normal distributions.
    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:

    Example 4 from :footcite:p:`wang2015conditional`.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z`` have shapes
        ``(n, 3)`` where `n` is the number of samples and `3` is the
        number of dimensions.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=1)
    check_in()

    rng = check_random_state(random_state)

    size = (n, 1)
    z = rng.normal(size=size)
    z1 = 0.5 * (np.power(z, 3) / 7 + z / 2)
    z2 = (np.power(z, 3) / 2 + z) / 3

    x1 = rng.normal(size=size)
    x2 = z1 + np.tanh(x1)
    x = x2 + np.power(x2, 3) / 3

    y1 = rng.normal(size=size)
    y2 = z2 + y1
    y = y2 + np.tanh(y2 / 3)

    return x, y, z


def independent_binomial(n, p, random_state=None):
    r"""
    Conditionally independent binomial distributions

    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}^p`:

    .. math::
        X_1, Y_1, Z_i &\sim \text{Binom}(10, 0.5) \\
        Z &= (Z_1, Z_2, \ldots, Z_p) \\
        X &= X_1 + Z_1 + Z_2 + \cdots + Z_p \\
        Y &= Y_1 + Z_1 + Z_2 + \cdots + Z_p

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions for conditioning variable ``Z``.

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x`` and ``y``, and ``z`` have shapes
        ``(n, 1)`` where `n` is the number of samples and `1` is the
        number of dimensions.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    rng = check_random_state(random_state)

    size = (n, 1)

    z = rng.binomial(10, 0.5, size=(n, p))
    x = rng.binomial(10, 0.5, size=size) + z.sum(axis=1)
    y = rng.binomial(10, 0.5, size=size) + z.sum(axis=1)

    return x, y, z


def correlated_normal(n, random_state=None):
    """
    Example 4 from :footcite:p:`szekelyPartialDistanceCorrelation2014a`
    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x`` and ``y``, and ``z`` have shapes
        ``(n, 1)`` where `n` is the number of samples and `1` is the
        number of dimensions.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=1)
    check_in()

    rng = check_random_state(random_state)

    mean = np.array([0, 0, 0])
    cov = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])

    x, y, z = rng.multivariate_normal(mean, cov, size=n).T

    return x, y, z


def correlated_lognormal(n, random_state=None):
    """
    Example 5 from :footcite:p:`szekelyPartialDistanceCorrelation2014a`
    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x`` and ``y``, and ``z`` have shapes
        ``(n, 1)`` where `n` is the number of samples and `1` is the
        number of dimensions.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=1)
    check_in()

    x, y, z = np.exp(correlated_normal(n, random_state))

    return x, y, z


def example9(n, p=4, random_state=None):
    """
    Example 9 from :footcite:p:`wang2015conditional`
    :math:`(X, Y, Z) \in \mathbb{R}^p \times \mathbb{R} \times \mathbb{R}`:

    .. math::
        Z_1, Z_2, \cdots, Z_p, Z_{p+1}, Z_{p+2}  &\sim t(1)\\
        X &= (Z_1, Z_2, Z_{p-1}, Z_p + Z_{p+1})\\
        Y &= Z_1 + Z_2 + \cdots + Z_p + Z_{p+1} + Z_{p+2}\\
        Z &= Z_{p+1}


    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x`` and ``y``, and ``z`` have shapes
        ``(n, 1)`` where `n` is the number of samples and `1` is the
        number of dimensions.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    rng = check_random_state(random_state)

    z_i = rng.standard_t(df=1, size=(n, p + 2))

    x = z_i[:, :p]
    x[:, -1] += z_i[:, p]
    y = z_i.sum(axis=1)
    z = z_i[:, p]

    return x, y, z


def example10(n, random_state=None):
    """
    Example 10 from :footcite:p:`wang2015conditional`
    :math:`(X, Y, Z) \in \mathbb{R}^10 \times \mathbb{R}^2 \times \mathbb{R}`:

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x`` and ``y``, and ``z`` have shapes
        ``(n, 1)`` where `n` is the number of samples and `1` is the
        number of dimensions.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=1)
    check_in()

    rng = check_random_state(random_state)

    z = rng.standard_t(df=1, size=(13, n))

    x = z[:11]
    x[9] += z[10]
    y1 = z[0] * z[1] + z[2] * z[3] + z[4] * z[10] + z[11]
    y2 = z[5] * z[6] + z[7] * z[8] + z[9] * z[10] + z[12]
    y = np.array([y1, y2])
    z = z[10]

    return x.T, y.T, z


def example11(n, random_state=None):
    """
    Example 11 from :footcite:p:`wang2015conditional`
    :math:`(X, Y, Z) \in \mathbb{R}^4 \times \mathbb{R}^2 \times \mathbb{R}^2`:

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x`` and ``y``, and ``z`` have shapes
        ``(n, 1)`` where `n` is the number of samples and `1` is the
        number of dimensions.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=1)
    check_in()

    rng = check_random_state(random_state)

    x = rng.standard_t(df=2, size=(4, n))
    y1 = np.sin(x[0]) + np.cos(x[1]) + x[2] ** 2 + x[3] ** 2
    y2 = x[0] ** 2 + x[1] ** 2 + x[2] + x[3]
    y = np.array([y1, y2])
    z = x[:2]

    return x.T, y.T, z.T


COND_SIMS = {}
