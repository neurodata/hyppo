import numpy as np


class _CheckInputs:
    def __init__(self, n, p, *args):
        self.n = n
        self.p = p

        self.extra_args = list(args)

    def __call__(self):
        for arg in self.extra_args:
            if type(arg[0]) is not arg[1]:
                raise ValueError("Incorrect input variable type")


def _gen_coeffs(p):
    """ Calculates coefficients polynomials """
    return np.array([1/(i+1) for i in range(p)]).reshape(-1, 1)


def _random_uniform(n, p, low=-1, high=1):
    return np.array(np.random.uniform(low, high, size=(n, p)))


def linear(n, p, noise=1, low=-1, high=1):
    r"""
    Simulates univariate or multivariate linear data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : float, (default: 1)
        The noise amplitude of the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Linear :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= w^T X + \kappa \epsilon

    Examples
    --------
    >>> from mgc.sims import linear
    >>> x, y = linear(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)

    """

    extra_type = [(noise, float),
                  (low, float),
                  (high, float)]
    _CheckInputs(n, p, *extra_type)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    gauss_noise = np.random.normal(0, 1, size=(n, 1))
    kappa = int(p == 1)
    y = x @ coeffs + kappa*noise*gauss_noise

    return x, y


def exponential(n, p, noise=10, low=0, high=3):
    r"""
    Simulates univariate or multivariate exponential data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : float, (default: 10)
        The noise amplitude of the simulation.
    low : float, (default: 0)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: 3)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Exponential :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(0, 3)^p \\
        Y &= \exp (w^T X) + 10 \kappa \epsilon

    Examples
    --------
    >>> from mgc.sims import exponential
    >>> x, y = exponential(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)

    """

    extra_type = [(noise, float),
                  (low, float),
                  (high, float)]
    _CheckInputs(n, p, *extra_type)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    gauss_noise = np.random.normal(0, 1, size=(n, 1))
    kappa = int(p == 1)
    y = np.exp(x @ coeffs) + kappa*noise*gauss_noise

    return x, y


def cubic(n, p, noise=80, low=-1, high=1, cubs=[-12, 48, 128], scale=1/3):
    r"""
    Simulates univariate or multivariate cubic data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : float, (default: 80)
        The noise amplitude of the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.
    cubs : list of ints (default: [-12, 48, 128])
        Coefficients of the cubic function where each value corresponds to the
        order of the cubic polynomial.
    scale : float (default: 1/3)
        Scaling center of the cubic.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Cubic :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= 128 \left( w^T X - \frac{1}{3} \right)^3
             + 48 \left( w^T X - \frac{1}{3} \right)^2
             - 12 \left( w^T X - \frac{1}{3} \right)
             + 80 \kappa \epsilon

    Examples
    --------
    >>> from mgc.sims import cubic
    >>> x, y = cubic(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)

    """

    extra_type = [(noise, float),
                  (low, float),
                  (high, float),
                  (cubs, list),
                  (scale, float)]
    _CheckInputs(n, p, *extra_type)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    gauss_noise = np.random.normal(0, 1, size=(n, 1))
    kappa = int(p == 1)

    x_coeffs = x @ coeffs
    y = (cubs[2] * (x_coeffs-scale)**3
         + cubs[1] * (x_coeffs-scale)**2
         + cubs[0] * (x_coeffs-scale)**3
         + kappa*noise*gauss_noise)

    return x, y


def spiral(n, p, noise=0.4, low=0, high=5):
    r"""
    Simulates univariate or multivariate spiral data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : int, (default: 0.4)
        The noise amplitude of the simulation.
    low : float, (default: 0)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: 5)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Spiral :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`: For
    :math:`U \sim \mathcal{U}(0, 5)`, :math:`\epsilon \sim \mathcal{N}(0, 1)`

    .. math::

        X_{|d|} &= U \sin(\pi U) \cos^d(\pi U)\ \mathrm{for}\ d = 1,...,p-1 \\
        X_{|p|} &= U \cos^p(\pi U) \\
        Y &= U \sin(\pi U) + 0.4 p \epsilon

    Examples
    --------
    >>> from mgc.sims import spiral
    >>> x, y = spiral(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)

    """

    extra_type = [(noise, float),
                  (low, float),
                  (high, float)]
    _CheckInputs(n, p, *extra_type)

    unif = _random_uniform(n, p=1, low=low, high=high)
    sinusoid = np.cos(np.pi * unif).reshape(-1, 1)
    y = unif * np.sin(np.pi * unif).reshape(-1, 1)

    x = np.zeros((n, p))
    if p > 1:
        for i in range(p-1):
            x[:, i] = np.squeeze(y * (sinusoid ** i))
    x[:, p-1] = np.squeeze(unif * sinusoid)

    guass_noise = np.random.normal(0, 1, size=(n, 1))
    y = y + noise*p*guass_noise

    return x, y
