import numpy as np
from sklearn.utils import check_random_state

from .indep_sim import _CheckInputs


def indep_normal(n, p=1, random_state=None):
    r"""
    Independent standard normal distributions.

    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:

    .. math::
        X, Y, Z &\sim N(0, 1)

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        Ignored.

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    rng = check_random_state(random_state)

    mean = np.array([0, 0, 0])
    cov = np.eye(3)

    x, y, z = rng.multivariate_normal(mean, cov, size=n).T

    return x, y, z


def indep_lognormal(n, p=1, random_state=None):
    r"""
    Independent lognormal and normal distributions.

    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:
    .. math::
        X &\sim \text{log} N(0, 1)\\
        Y, Z &\sim N(0, 1)

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    x, y, z = indep_normal(n=n, p=p, random_state=random_state)
    x = np.exp(x)

    return x, y, z


def indep_binomial(n, p=1, random_state=None):
    r"""
    Independent binomial distributions.

    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}^p`:
    .. math::
        X, Y, Z_i &\sim \text{Binom}(10, 0.5) \\
        Z &= (Z_1, Z_2, \ldots, Z_p)

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions for conditioning variable ``Z``.

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x`` and ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    rng = check_random_state(random_state)

    size = (n, 1)

    z = rng.binomial(10, 0.5, size=(n, p))
    x = rng.binomial(10, 0.5, size=size)
    y = rng.binomial(10, 0.5, size=size)

    return x, y, z


def cond_indep_normal(n, p=1, random_state=None):
    r"""
    Conditionally independent normal distributions.

    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:
    .. math::
        \mu &= (0, 0, 0)\\
        \Sigma &= \begin{bmatrix}
        1 & 0.36 & 0.6 \\
        0.36 & 1 & 0.6 \\
        0.6 & 0.6 & 1
        \end{bmatrix}\\
        (X, Y, Z) &\sim MVN(\mu, \Sigma)

    The conditional covariance matrix is given by:
    .. math::
        \Sigma(X, Y | Z) &= \begin{bmatrix}
        0.64 & 0\\
        0 & 0.64
        \end{bmatrix}

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        Ignored.

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    rng = check_random_state(random_state)

    mean = np.array([0, 0, 0])
    cov = np.array([[1, 0.36, 0.6], [0.36, 1, 0.6], [0.6, 0.6, 1]])

    x, y, z = rng.multivariate_normal(mean, cov, size=n).T

    return x, y, z


def cond_indep_lognormal(n, p=1, random_state=None):
    r"""
    Conditionally independent lognormal and normal distributions.

    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        Ignored.

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    rng = check_random_state(random_state)

    mean = np.array([0, 0, 0])
    cov = np.array([[1, 0.36, 0.6], [0.36, 1, 0.6], [0.6, 0.6, 1]])

    x, y, z = rng.multivariate_normal(mean, cov, size=n).T
    x = np.exp(x)

    return x, y, z


def cond_indep_normal_nonlinear(n, p=1, random_state=None):
    r"""
    Conditionally independent normal distributions. Example 3 from :footcite:p:`wang2015conditional`.

    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:
    .. math::
        X_1, Y_1, Z &\sim N(0, 1) \\
        Z_1 &= 0.5 \left( \frac{Z^3}{7} + \frac{Z}{2} \right) \\
        Z_2 &= \frac{Z^3}{2} + \frac{Z}{3} \\
        X_2 &= Z_1 + \tanh(X_1) \\
        X &= X_2 + \frac{X_2^3}{3}\\
        Y_2 &= Z_2 + Y_1 \\
        Y &= Y_2 + \frac{Y_2^3}{3}

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
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


def cond_indep_binomial(n, p=1, random_state=None):
    r"""
    Conditionally independent binomial distributions.

    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}^p`:

    .. math::
        X_1, Y_1, Z_i &\sim \text{Binom}(10, 0.5) \\
        Z &= (Z_1, Z_2, \ldots, Z_p) \\
        X &= X_1 + Z_1 + Z_2 + \cdots + Z_p \\
        Y &= Y_1 + Z_1 + Z_2 + \cdots + Z_p

    Examples 2 and 4 from :footcite:p:`wang2015conditional`.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions for conditioning variable ``Z``.

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    rng = check_random_state(random_state)

    size = (n, 1)

    z = rng.binomial(10, 0.5, size=(n, p))
    x = rng.binomial(10, 0.5, size=size) + z.sum(axis=1, keepdims=True)
    y = rng.binomial(10, 0.5, size=size) + z.sum(axis=1, keepdims=True)

    return x, y, z


def correlated_binomial(n, p=1, random_state=None):
    r"""
    Conditionally dependent binomial distributions. Examples 6 from:footcite:p:`wang2015conditional`.

    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:

    .. math::
        X_1, Z &\sim \text{Binom}(10, 0.5) \\
        X &= X_1 + Z \\
        Y &= (X_1 - 5)^4 + Z


    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions for conditioning variable ``Z``.

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    rng = check_random_state(random_state)

    x1, z = rng.binomial(10, 0.5, size=(2, n))

    x = x1 + z
    y = (x1 - 5) ** 4 + z

    return x, y, z


def correlated_normal(n, p=1, random_state=None):
    r"""
    Conditionally dependent normal distributions. Example 5 from :footcite:p:`wang2015conditional`

    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:
    .. math::
        \mu &= (0, 0, 0)\\
        \Sigma &= \begin{bmatrix}
        1 & 0.7 & 0.6\\
        0.7 & 1 & 0.6\\
        0.6 & 0.6 & 1
        \end{bmatrix}\\
        (X, Y, Z) &\sim MVN(\mu, \Sigma)

    The conditional covariance matrix is given by:
    .. math::
        \Sigma(X, Y | Z) &= \begin{bmatrix}
        0.64 & 0.34\\
        0.34 & 0.64
        \end{bmatrix}

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    rng = check_random_state(random_state)

    mean = np.array([0, 0, 0])
    cov = np.array([[1, 0.7, 0.6], [0.7, 1, 0.6], [0.6, 0.6, 1]])

    x, y, z = rng.multivariate_normal(mean, cov, size=n).T

    return x, y, z


def correlated_normal_nonliear(n, p=1, random_state=None):
    r"""
    Conditionally dependent normal distributions with nonlinear dependence.
    Example 7 from :footcite:p:`wang2015conditional`

    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:
    .. math::
        X_1, Y_1, Z, \epsilon &\sim N(0, 1) \\
        Z_1 &= 0.5(Z^3/7 + Z/2) \\
        Z_2 &= (Z^3/2 + Z)/3 \\
        X_2 &= Z_1 + \tanh(X_1) \\
        X_3 &= X_2 + X_2^3 / 3 \\
        Y_2 &= Z_2 + Y_1\\
        Y_3 &= Y_2 + \tanh(Y_2 / 3) \\
        X &= X_3 + \cosh\epsilon \\
        Y &= Y_3 + \cosh\epsilon^2

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    rng = check_random_state(random_state)

    x1, y1, z, eps = rng.normal(size=(n, 4)).T

    z1 = 0.5 * (z**3 / 7 + z / 2)
    z2 = (z**3 / 2 + z) / 3

    x2 = z1 + np.tanh(x1)
    x3 = x2 + x2**3 / 3

    y2 = z2 + y1
    y3 = y2 + np.tanh(y2 / 3)

    x = x3 + np.cosh(eps)
    y = y3 + np.cosh(eps**2)

    return x, y, z


def correlated_lognormal(n, p=1, random_state=None):
    r"""
    Example 5 from :footcite:p:`szekelyPartialDistanceCorrelation2014a`
    :math:`(X, Y, Z) \in \mathbb{R} \times \mathbb{R} \times \mathbb{R}`:

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    x, y, z = correlated_normal(n, p, random_state)
    x = np.exp(x)

    return x, y, z


def correlated_t_linear(n, p=4, random_state=None):
    r"""
    Conditionally dependent t-distributed data with linear dependence.
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
    p : int
        The number of dimentions for variable ``x``.

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

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


def correlated_t_quadratic(n, p=10, random_state=None):
    r"""
    Conditionally dependent t-distributed data with quadratic dependence.
    Example 10 from :footcite:p:`wang2015conditional`

    :math:`(X, Y, Z) \in \mathbb{R}^10 \times \mathbb{R}^2 \times \mathbb{R}`:
    .. math::
        Z_1, Z_2, \cdots, Z_{13}  &\sim t(1)\\
        X_i &= Z_i, i = 1, 2, \cdots, 9\\
        X_{10} &= Z_{10} + Z_{11}\\
        Y_1 &= Z_1 Z_2 + Z_3 Z_4 + Z_5 Z_{11} + Z_{12}\\
        Y_2 &= Z_6 Z_7 + Z_8 Z_9 + Z_{10} Z_{11} + Z_{13}\\
        Z &= Z_{11}

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
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


def correlated_t_nonlinear(n, p=4, random_state=None):
    r"""
    Conditionally dependent t-distributed data with nonlinear dependence.
    Example 11 from :footcite:p:`wang2015conditional`

    :math:`(X, Y, Z) \in \mathbb{R}^4 \times \mathbb{R}^2 \times \mathbb{R}^2`:
    .. math::
        Z_1, \ldots, Z_4 &\sim t(2)\\
        Y_1 &= \sin(Z_1) + \cos(Z_2) + Z_3^2 + Z_4^2\\
        Y_2 &= Z_1^2 + Z_2^2 + Z_3 + Z_4\\
        X &= (Z_1, Z_2, Z_3, Z_4)\\
        Y &= (Y_1, Y_2)\\
        Z &= (Z_1, Z_2)

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).

    Returns
    -------
    x,y,z : ndarray of float
        Simulated data matrices. ``x``, ``y``, and ``z``.

    References
    ----------
    .. footbibliography::
    """
    check_in = _CheckInputs(n, p=p)
    check_in()

    rng = check_random_state(random_state)

    z = rng.standard_t(df=2, size=(4, n))
    y1 = np.sin(z[0]) + np.cos(z[1]) + z[2] ** 2 + z[3] ** 2
    y2 = z[0] ** 2 + z[1] ** 2 + z[2] + z[3]
    y = np.array([y1, y2])

    return z.T, y.T, z[:2].T


COND_SIMULATIONS = {
    "independent_normal": indep_normal,
    "independent_lognormal": indep_lognormal,
    "independent_binomial": indep_binomial,
    "cond_independent_normal": cond_indep_normal,
    "cond_independent_lognormal": cond_indep_lognormal,
    "cond_independent_binomial": cond_indep_binomial,
    "cond_independent_normal_nonlinear": cond_indep_normal_nonlinear,
    "correlated_normal": correlated_normal,
    "correlated_lognormal": correlated_lognormal,
    "correlated_binomial": correlated_binomial,
    "correlated_normal_nonliear": correlated_normal_nonliear,
    "correlated_t_linear": correlated_t_linear,
    "correlated_t_nonlinear": correlated_t_nonlinear,
    "correlated_t_quadratic": correlated_t_quadratic,
}


def condi_indep_sim(sim, n, p, random_state=None, **kwargs):
    r"""
    Conditional independence simulation generator.

    Takes a simulation and the required parameters, and outputs the simulated
    data matrices.

    Parameters
    ----------
    sim : str
        The name of the simulation (from the :mod:`hyppo.tools` module) that is to be
        rotated.
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    **kwargs
        Additional keyword arguements for the desired simulation.

    Returns
    -------
    x,y, z : ndarray of float
        Simulated data matrices.
    """
    if sim not in COND_SIMULATIONS.keys():
        raise ValueError(
            "sim_name must be one of the following: {}".format(
                list(COND_SIMULATIONS.keys())
            )
        )
    else:
        sim = COND_SIMULATIONS[sim]

    x, y, z = sim(n, p, random_state, **kwargs)

    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    if z.ndim == 1:
        z = z[:, np.newaxis]

    return x, y, z
