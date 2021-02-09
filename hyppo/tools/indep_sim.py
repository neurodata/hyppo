import numpy as np


class _CheckInputs:
    """Check if additional arguments are correct."""

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def __call__(self, **kwargs):
        if type(self.n) is not int or type(self.p) is not int:
            raise ValueError(
                "Expected n and p of type int, got {} and {}".format(
                    type(self.n), type(self.p)
                )
            )

        if self.n < 5:
            raise ValueError("n must be >= 5, got {}".format(self.n))

        if self.p < 1:
            raise ValueError("p must be >= 1, got {}".format(self.p))

        for key, value in kwargs.items():
            if value[1] is float and type(value[0]) is int:
                continue
            if type(value[0]) is not value[1]:
                raise ValueError(
                    "Expected {} type {} got {}".format(key, value[1], type(value[0]))
                )


def _gen_coeffs(p):
    """Calculates coefficients polynomials."""
    return np.array([1 / (i + 1) for i in range(p)]).reshape(-1, 1)


def _random_uniform(n, p, low=-1, high=1):
    """Generate random uniform data."""
    return np.array(np.random.uniform(low, high, size=(n, p)))


def _calc_eps(n):
    """Calculate noise."""
    return np.random.normal(0, 1, size=(n, 1))


def linear(n, p, noise=False, low=-1, high=1):
    r"""
    Linear simulation.

    Linear :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= w^T X + \kappa \epsilon

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: -1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, 1)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    extra_args = {
        "noise": (noise, bool),
        "low": (low, float),
        "high": (high, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)
    y = x @ coeffs + 1 * noise * eps

    return x, y


def exponential(n, p, noise=False, low=0, high=3):
    r"""
    Exponential simulation.

    Exponential :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(0, 3)^p \\
        Y &= \exp (w^T X) + 10 \kappa \epsilon

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: 0
        The lower limit of the uniform distribution simulated from.
    high : float, default: 3
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, 1)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    extra_args = {
        "noise": (noise, bool),
        "low": (low, float),
        "high": (high, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)
    y = np.exp(x @ coeffs) + 10 * noise * eps

    return x, y


def cubic(n, p, noise=False, low=-1, high=1, cubs=[-12, 48, 128], scale=1 / 3):
    r"""
    Cubic simulation.

    Cubic :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= 128 \left( w^T X - \frac{1}{3} \right)^3
             + 48 \left( w^T X - \frac{1}{3} \right)^2
             - 12 \left( w^T X - \frac{1}{3} \right)
             + 80 \kappa \epsilon

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: -1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.
    cubs : list of ints, default: [-12, 48, 128]
        Coefficients of the cubic function where each value corresponds to the
        order of the cubic polynomial.
    scale : float, default: 1/3
        Scaling center of the cubic.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, 1)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    extra_args = {
        "noise": (noise, bool),
        "low": (low, float),
        "high": (high, float),
        "cubs": (cubs, list),
        "scale": (scale, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)

    x_coeffs = x @ coeffs - scale
    y = (
        cubs[2] * x_coeffs ** 3
        + cubs[1] * x_coeffs ** 2
        + cubs[0] * x_coeffs ** 3
        + 80 * noise * eps
    )

    return x, y


def joint_normal(n, p, noise=False):
    r"""
    Joint Normal simulation.

    Joint Normal :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`: Let
    :math:`\rho = \frac{1}{2} p`, :math:`I_p` be the identity matrix of size
    :math:`p \times p`, :math:`J_p` be the matrix of ones of size
    :math:`p \times p` and
    :math:`\Sigma = \begin{bmatrix} I_p & \rho J_p \\ \rho J_p & (1 + 0.5\kappa) I_p \end{bmatrix}`.
    Then,

    .. math::

        (X, Y) \sim \mathcal{N}(0, \Sigma)

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, p)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    if p > 10:
        raise ValueError("Covariance matrix for p > 10 is not positive semi-definite")

    extra_args = {
        "noise": (noise, bool),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    rho = 1 / (2 * p)
    cov1 = np.concatenate((np.identity(p), rho * np.ones((p, p))), axis=1)
    cov2 = np.concatenate((rho * np.ones((p, p)), np.identity(p)), axis=1)
    covT = np.concatenate((cov1.T, cov2.T), axis=1)

    eps = _calc_eps(n)
    x = np.random.multivariate_normal(np.zeros(2 * p), covT, n)
    y = x[:, p : 2 * p] + 0.5 * noise * eps
    x = x[:, :p]

    return x, y


def step(n, p, noise=False, low=-1, high=1):
    r"""
    Step simulation.

    Step :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= \mathbb{1}_{w^T X > 0} + \epsilon

    where :math:`\mathbb{1}` is the indicator function.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: -1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, 1)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    extra_args = {
        "noise": (noise, bool),
        "low": (low, float),
        "high": (high, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    if p > 1:
        noise = True
    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)

    x_coeff = ((x @ coeffs) > 0) * 1
    y = x_coeff + noise * eps

    return x, y


def quadratic(n, p, noise=False, low=-1, high=1):
    r"""
    Quadratic simulation.

    Quadratic :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= (w^T X)^2 + 0.5 \kappa \epsilon

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: -1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, 1)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    extra_args = {
        "noise": (noise, bool),
        "low": (low, float),
        "high": (high, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)

    x_coeffs = x @ coeffs
    y = x_coeffs ** 2 + 0.5 * noise * eps

    return x, y


def w_shaped(n, p, noise=False, low=-1, high=1):
    r"""
    W-Shaped simulation.

    W-Shaped :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:
    :math:`\mathcal{U}(-1, 1)^p`,

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= \left[ \left( (w^T X)^2 - \frac{1}{2} \right)^2
                            + \frac{w^T U}{500} \right] + 0.5 \kappa \epsilon

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: -1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, 1)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    extra_args = {
        "noise": (noise, bool),
        "low": (low, float),
        "high": (high, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    x = _random_uniform(n, p, low, high)
    u = _random_uniform(n, p, 0, 1)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)

    x_coeffs = x @ coeffs
    u_coeffs = u @ coeffs
    y = 4 * ((x_coeffs ** 2 - 0.5) ** 2 + u_coeffs / 500) + 0.5 * noise * eps

    return x, y


def spiral(n, p, noise=False, low=0, high=5):
    r"""
    Spiral simulation.

    Spiral :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:
    :math:`U \sim \mathcal{U}(0, 5)`, :math:`\epsilon \sim \mathcal{N}(0, 1)`

    .. math::

        X_{|d|} &= U \sin(\pi U) \cos^d(\pi U)\ \mathrm{for}\ d = 1,...,p-1 \\
        X_{|p|} &= U \cos^p(\pi U) \\
        Y &= U \sin(\pi U) + 0.4 p \epsilon

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: 0
        The lower limit of the uniform distribution simulated from.
    high : float, default: 5
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, 1)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    extra_args = {
        "noise": (noise, bool),
        "low": (low, float),
        "high": (high, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    if p > 1:
        noise = True
    rx = _random_uniform(n, p=1, low=low, high=high)
    ry = rx
    rx = np.repeat(rx, p, axis=1)
    z = rx
    x = np.zeros((n, p))
    x[:, 0] = np.cos(z[:, 0] * np.pi)
    for i in range(p - 1):
        x[:, i + 1] = np.multiply(x[:, i], np.cos(z[:, i + 1] * np.pi))
        x[:, i] = np.multiply(x[:, i], np.sin(z[:, i + 1] * np.pi))
    x = np.multiply(rx, x)
    y = np.multiply(ry, np.sin(z[:, 0].reshape(-1, 1) * np.pi))

    eps = _calc_eps(n)
    y = y + 0.4 * p * noise * eps

    return x, y


def uncorrelated_bernoulli(n, p, noise=False, prob=0.5):
    r"""
    Uncorrelated Bernoulli simulation.

    Uncorrelated Bernoulli :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:
    :math:`U \sim \mathcal{B}(0.5)`, :math:`\epsilon_1 \sim \mathcal{N}(0, I_p)`,
    :math:`\epsilon_2 \sim \mathcal{N}(0, 1)`,

    .. math::

        X &= \mathcal{B}(0.5)^p + 0.5 \epsilon_1 \\
        Y &= (2U - 1) w^T X + 0.5 \epsilon_2

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    prob : float, default: 0.5
        The probability of the bernoulli distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, 1)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    extra_args = {
        "noise": (noise, bool),
        "prob": (prob, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    binom = np.random.binomial(1, prob, size=(n, 1))
    sig = np.identity(p)
    gauss_noise = np.random.multivariate_normal(np.zeros(p), sig, size=n)

    x = np.random.binomial(1, prob, size=(n, p)) + 0.5 * noise * gauss_noise
    coeffs = _gen_coeffs(p)

    eps = _calc_eps(n)
    x_coeffs = x @ coeffs
    y = binom * 2 - 1
    y = np.multiply(x_coeffs, y) + 0.5 * noise * eps

    return x, y


def logarithmic(n, p, noise=False):
    r"""
    Logarithmic simulation.

    Logarithmic :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`\epsilon \sim \mathcal{N}(0, I_p)`,

    .. math::

        X &\sim \mathcal{N}(0, I_p) \\
        Y_{|d|} &= 2 \log_2 (|X_{|d|}|) + 3 \kappa \epsilon_{|d|}
                   \ \mathrm{for}\ d = 1, ..., p

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, p)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    extra_args = {
        "noise": (noise, bool),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    sig = np.identity(p)
    x = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    eps = _calc_eps(n)

    y = np.log(x ** 2) + 3 * noise * eps

    return x, y


def fourth_root(n, p, noise=False, low=-1, high=1):
    r"""
    Fourth Root simulation.

    Fourth Root :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= |w^T X|^\frac{1}{4} + \frac{\kappa}{4} \epsilon

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: -1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, 1)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    extra_args = {
        "noise": (noise, bool),
        "low": (low, float),
        "high": (high, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    x = _random_uniform(n, p, low, high)
    eps = _calc_eps(n)
    coeffs = _gen_coeffs(p)

    x_coeffs = x @ coeffs
    y = np.abs(x_coeffs) ** 0.25 + 0.25 * noise * eps

    return x, y


def _sin(n, p, noise=False, low=-1, high=1, period=4 * np.pi):
    """Helper function to calculate sine simulation"""
    extra_args = {
        "noise": (noise, bool),
        "low": (low, float),
        "high": (high, float),
        "period": (period, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    x = _random_uniform(n, p, low, high)
    if p > 1 or noise:
        sig = np.identity(p)
        v = np.random.multivariate_normal(np.zeros(p), sig, size=n)
        x = x + 0.02 * p * v
    eps = _calc_eps(n)

    if period == 4 * np.pi:
        cc = 1
    else:
        cc = 0.5

    y = np.sin(x * period) + cc * noise * eps

    return x, y


def sin_four_pi(n, p, noise=False, low=-1, high=1):
    r"""
    Sine 4\ :math:`\pi` simulation.

    Sine 4\ :math:`\pi` :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{U}(-1, 1)`, :math:`V \sim \mathcal{N}(0, 1)^p`,
    :math:`\theta = 4 \pi`,

    .. math::

        X_{|d|} &= U + 0.02 p V_{|d|}\ \mathrm{for}\ d = 1, ..., p \\
        Y &= \sin (\theta X) + \kappa \epsilon

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: -1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, p)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    return _sin(n, p, noise=noise, low=low, high=high, period=4 * np.pi)


def sin_sixteen_pi(n, p, noise=False, low=-1, high=1):
    r"""
    Sine 16\ :math:`\pi` simulation.

    Sine 16\ :math:`\pi` :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{U}(-1, 1)`, :math:`V \sim \mathcal{N}(0, 1)^p`,
    :math:`\theta = 16 \pi`,

    .. math::

        X_{|d|} &= U + 0.02 p V_{|d|}\ \mathrm{for}\ d = 1, ..., p \\
        Y &= \sin (\theta X) + \kappa \epsilon

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: 1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, p)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    return _sin(n, p, noise=noise, low=low, high=high, period=16 * np.pi)


def _square_diamond(n, p, noise=False, low=-1, high=1, period=-np.pi / 2):
    """Helper function to calculate square/diamond simulation"""
    extra_args = {
        "noise": (noise, bool),
        "low": (low, float),
        "high": (high, float),
        "period": (period, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    u = _random_uniform(n, p, low, high)
    v = _random_uniform(n, p, low, high)
    sig = np.identity(p)
    gauss_noise = np.random.multivariate_normal(np.zeros(p), sig, size=n)

    x = u * np.cos(period) + v * np.sin(period) + 0.05 * p * gauss_noise
    y = -u * np.sin(period) + v * np.cos(period)

    return x, y


def square(n, p, noise=False, low=-1, high=1):
    r"""
    Square simulation.

    Square :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{U}(-1, 1)`, :math:`V \sim \mathcal{N}(0, 1)^p`,
    :math:`\theta = -\frac{\pi}{8}`,

    .. math::

        X_{|d|} &= U \cos(\theta) + V \sin(\theta) + 0.05 p \epsilon_{|d|}
        \ \mathrm{for}\ d = 1, ..., p \\
        Y_{|d|} &= -U \sin(\theta) + V \cos(\theta)

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: -1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, p)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    return _square_diamond(n, p, noise=noise, low=low, high=high, period=-np.pi / 8)


def two_parabolas(n, p, noise=False, low=-1, high=1, prob=0.5):
    r"""
    Two Parabolas simulation.

    Two Parabolas :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= ((w^T X)^2 + 2 \kappa \epsilon) \times \left( U = \frac{1}{2} \right)

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: -1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.
    prob : float, default: 0.5
        The probability of the bernoulli distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, 1)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    extra_args = {
        "noise": (noise, bool),
        "low": (low, float),
        "high": (high, float),
        "prob": (prob, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    u = np.random.binomial(1, prob, size=(n, 1))
    rand_noise = _random_uniform(n, p, low=0, high=1)

    x_coeffs = x @ coeffs
    y = (x_coeffs ** 2 + 2 * noise * rand_noise) * (u - 0.5)

    return x, y


def _circle_ellipse(n, p, noise=False, low=-1, high=1, radius=1):
    """Helper function to calculate circle/ellipse simulation"""
    extra_args = {
        "noise": (noise, bool),
        "low": (low, float),
        "high": (high, float),
        "radius": (radius, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    if p > 1:
        noise = True
    x = _random_uniform(n, p, low, high)
    rx = radius * np.ones((n, p))
    unif = _random_uniform(n, p, low, high)
    sig = np.identity(p)
    gauss_noise = np.random.multivariate_normal(np.zeros(p), sig, size=n)

    ry = np.ones((n, p))
    x[:, 0] = np.cos(unif[:, 0] * np.pi)
    for i in range(p - 1):
        x[:, i + 1] = x[:, i] * np.cos(unif[:, i + 1] * np.pi)
        x[:, i] = x[:, i] * np.sin(unif[:, i + 1] * np.pi)

    x = rx * x + 0.4 * noise * rx * gauss_noise
    y = ry * np.sin(unif[:, 0].reshape(n, 1) * np.pi)

    return x, y


def circle(n, p, noise=False, low=-1, high=1):
    r"""
    Circle simulation.

    Circle :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{U}(-1, 1)^p`, :math:`\epsilon \sim \mathcal{N}(0, I_p)`,
    :math:`r = 1`,

    .. math::

        X_{|d|} &= r \left( \sin(\pi U_{|d+1|}) \prod_{j=1}^d \cos(\pi U_{|j|})
        + 0.4 \epsilon_{|d|} \right)\ \mathrm{for}\ d = 1, ..., p-1 \\
        X_{|p|} &= r \left( \prod_{j=1}^p \cos(\pi U_{|j|})
        + 0.4 \epsilon_{|p|} \right) \\
        Y_{|d|} &= \sin(\pi U_{|1|})

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: -1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, p)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    return _circle_ellipse(n, p, noise=noise, low=low, high=high, radius=1)


def ellipse(n, p, noise=False, low=-1, high=1):
    r"""
    Ellipse simulation.

    Ellipse :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{U}(-1, 1)^p`, :math:`\epsilon \sim \mathcal{N}(0, I_p)`,
    :math:`r = 5`,

    .. math::

        X_{|d|} &= r \left( \sin(\pi U_{|d+1|}) \prod_{j=1}^d \cos(\pi U_{|j|})
        + 0.4 \epsilon_{|d|} \right)\ \mathrm{for}\ d = 1, ..., p-1 \\
        X_{|p|} &= r \left( \prod_{j=1}^p \cos(\pi U_{|j|})
        + 0.4 \epsilon_{|p|} \right) \\
        Y_{|d|} &= \sin(\pi U_{|1|})

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: -1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, p)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    return _circle_ellipse(n, p, noise=noise, low=low, high=high, radius=5)


def diamond(n, p, noise=False, low=-1, high=1):
    r"""
    Diamond simulation.

    Diamond :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{U}(-1, 1)`, :math:`V \sim \mathcal{N}(0, 1)^p`,
    :math:`\theta = -\frac{\pi}{4}`,

    .. math::

        X_{|d|} &= U \cos(\theta) + V \sin(\theta)
        + 0.05 p \epsilon_{|d|}\ \mathrm{for}\ d = 1, ..., p \\
        Y_{|d|} &= -U \sin(\theta) + V \cos(\theta)

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    noise : bool, default: False
        Whether or not to include noise in the simulation.
    low : float, default: -1
        The lower limit of the uniform distribution simulated from.
    high : float, default: 1
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, p)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    return _square_diamond(n, p, noise=noise, low=low, high=high, period=-np.pi / 4)


def multiplicative_noise(n, p):
    r"""
    Multiplicative Noise simulation.

    Multiplicative Noise :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`\U \sim \mathcal{N}(0, I_p)`,

    .. math::

        X &\sim \mathcal{N}(0, I_p) \\
        Y_{|d|} &= U_{|d|} X_{|d|}\ \mathrm{for}\ d = 1, ..., p

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, p)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    check_in = _CheckInputs(n, p)
    check_in()

    sig = np.identity(p)
    x = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    y = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    y = np.multiply(x, y)

    return x, y


def multimodal_independence(n, p, prob=0.5, sep1=3, sep2=2):
    r"""
    Multimodal Independence data.

    Multimodal Independence :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{N}(0, I_p)`, :math:`V \sim \mathcal{N}(0, I_p)`,
    :math:`U^\prime \sim \mathcal{B}(0.5)^p`, :math:`V^\prime \sim \mathcal{B}(0.5)^p`,

    .. math::

        X &= \frac{U}{3} + 2 U^\prime - 1 \\
        Y &= \frac{V}{3} + 2 V^\prime - 1

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    prob : float, default: 0.5
        The probability of the bernoulli distribution simulated from.
    sep1, sep2: float, default: 3, 2
        The separation between clusters of normally distributed data.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices. ``x` and ``y`` have shapes ``(n, p)`` and ``(n, p)``
        where `n` is the number of samples and `p` is the number of
        dimensions.
    """
    extra_args = {
        "prob": (prob, float),
        "sep1": (sep1, float),
        "sep2": (sep2, float),
    }
    check_in = _CheckInputs(n, p)
    check_in(**extra_args)

    sig = np.identity(p)
    u = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    v = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    u_2 = np.random.binomial(1, prob, size=(n, p))
    v_2 = np.random.binomial(1, prob, size=(n, p))

    x = u / sep1 + sep2 * u_2 - 1
    y = v / sep1 + sep2 * v_2 - 1

    return x, y


SIMULATIONS = {
    "linear": linear,
    "exponential": exponential,
    "cubic": cubic,
    "joint_normal": joint_normal,
    "step": step,
    "quadratic": quadratic,
    "w_shaped": w_shaped,
    "spiral": spiral,
    "uncorrelated_bernoulli": uncorrelated_bernoulli,
    "logarithmic": logarithmic,
    "fourth_root": fourth_root,
    "sin_four_pi": sin_four_pi,
    "sin_sixteen_pi": sin_sixteen_pi,
    "square": square,
    "two_parabolas": two_parabolas,
    "circle": circle,
    "ellipse": ellipse,
    "diamond": diamond,
    "multiplicative_noise": multiplicative_noise,
    "multimodal_independence": multimodal_independence,
}


def indep_sim(sim, n, p, **kwargs):
    r"""
    Independence simulation generator.

    Takes a simulation and the required parameters, and outputs the simulated
    data matrices.

    Parameters
    ----------
    sim : str
        The name of the simulation (from the :mod:`hyppo.tools module) that is to be
        rotated.
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    **kwargs
        Additional keyword arguements for the desired simulation.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices.
    """
    if sim not in SIMULATIONS.keys():
        raise ValueError(
            "sim_name must be one of the following: {}".format(list(SIMULATIONS.keys()))
        )
    else:
        sim = SIMULATIONS[sim]

    return sim(n, p, **kwargs)
