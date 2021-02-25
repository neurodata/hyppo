import numpy as np

from .indep_sim import SIMULATIONS


def _2samp_rotate(sim, x, y, p, degree=90, pow_type="samp"):
    """Generate an independence simulation, rotate it to produce another."""
    angle = np.radians(degree)
    data = np.hstack([x, y])
    same_shape = [
        "joint_normal",
        "logarithmic",
        "sin_four_pi",
        "sin_sixteen_pi",
        "two_parabolas",
        "square",
        "diamond",
        "circle",
        "ellipse",
        "multiplicative_noise",
        "multimodal_independence",
    ]
    if sim in same_shape:
        rot_shape = 2 * p
    else:
        rot_shape = p + 1
    rot_mat = np.identity(rot_shape)
    if pow_type == "dim":
        if sim not in [
            "exponential",
            "cubic",
            "spiral",
            "uncorrelated_bernoulli",
            "fourth_root",
            "circle",
        ]:
            for i in range(rot_shape):
                mat = np.random.normal(size=(rot_shape, 1))
                mat = mat / np.sqrt(np.sum(mat ** 2))
                if i == 0:
                    rot = mat
                else:
                    rot = np.hstack([rot, mat])
                rot_mat, _ = np.linalg.qr(rot)
                if (p % 2) == 1:
                    rot_mat[0] *= -1
        else:
            rot_mat[np.ix_((0, -1), (0, -1))] = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
    elif pow_type == "samp":
        rot_mat[np.ix_((0, 1), (0, 1))] = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
    else:
        raise ValueError("pow_type not a valid flag ('dim', 'samp')")
    rot_data = (rot_mat @ data.T).T

    if sim in same_shape:
        x_rot, y_rot = np.hsplit(rot_data, 2)
    else:
        x_rot, y_rot = np.hsplit(rot_data, [-p])

    return x_rot, y_rot


def rot_ksamp(sim, n, p, k=2, noise=True, degree=90, pow_type="samp", **kwargs):
    r"""
    Rotates input simulations to produce a `k`-sample simulation.

    Parameters
    ----------
    sim : str
        The name of the simulation (from the :mod:`hyppo.tools` module) that is to be
        rotated.
    n : int
        The number of samples desired by ``sim`` (>= 5).
    p : int
        The number of dimensions desired by ``sim`` (>= 1).
    k : int, default: 2
        The number of groups to simulate.
    noise : bool, default: True
        Whether or not to include noise in the simulation.
    degree : float or list of float, default: 90
        The number of degrees to rotate the input simulation by (in first dimension).
        The list must be the same size as ``k - 1``.
    pow_type : "samp", "dim", default: "samp"
        Simulation type, (increasing sample size or dimension).
    **kwargs
        Additional keyword arguments for the independence simulation.

    Returns
    -------
    sims : list of ndarray
        Rotated data matrices. ``sims`` is a list of arrays of shape ``(n, p+1)``
        or ``(n, 2p)`` depending on the independence simulation. Here, `n`
        is the number of samples and `p` is the number of dimensions.
    """
    if sim not in SIMULATIONS.keys():
        raise ValueError(
            "Not valid simulation, must be one of {}".format(SIMULATIONS.keys())
        )

    if (k - 1) > 1:
        if type(degree) is list:
            if (k - 1) != len(degree):
                raise ValueError(
                    "k={}, so length of degree must be {}, got {}".format(
                        k, k - 1, len(degree)
                    )
                )
        else:
            if (k - 1) != 1:
                raise ValueError(
                    "k={}, so degree must be list of length {}, got {}".format(
                        k, k - 1, type(degree)
                    )
                )

    if sim == "multimodal_independence":
        sims = [np.hstack(SIMULATIONS[sim](n, p, **kwargs)) for _ in range(k)]
    else:
        if sim != "multiplicative_noise":
            kwargs["noise"] = noise
        x, y = SIMULATIONS[sim](n, p, **kwargs)
        if (k - 1) == 1:
            sims = [
                np.hstack([x, y]),
                np.hstack(
                    _2samp_rotate(sim, x, y, p, degree=degree, pow_type=pow_type)
                ),
            ]
        else:
            sims = [np.hstack([x, y])] + [
                np.hstack(_2samp_rotate(sim, x, y, p, degree=deg, pow_type=pow_type))
                for deg in degree
            ]

    return sims


def gaussian_3samp(n, epsilon=1, weight=0, case=1):
    r"""
    Generates 3 sample of gaussians corresponding to 5 cases.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation (>= 5).
    epsilon : float, default: 1
        The amount to translate simulation by (amount  depends on case).
    weight : float, default: False
        Number between 0 and 1 corresponding to weight of the second Gaussian
        (used in case 4 and 5 to produce a mixture of Gaussians).
    case : 1, 2, 3, 4, 5, default: 1
        The case in which to evaluate statistical power for each test.

    Returns
    -------
    sims : list of ndarray
        List of 3 2-dimensional multivariate Gaussian each
        corresponding to the desired case.
    """
    if n < 5:
        raise ValueError("n must be >= 5, got {}".format(n))

    old_case = case
    if old_case == 4:
        case = 2
    elif old_case == 5:
        case = 3
    sigma = np.identity(2)
    mu1 = [0] * 3
    mu2 = [0] * 3

    if case == 1:
        pass
    elif case == 2:
        mu2 = [0, 0, epsilon]
    elif case == 3:
        mu1 = [0, -epsilon / 2, epsilon / 2]
        mu2 = [
            (np.sqrt(3) / 3) * epsilon,
            -(np.sqrt(3) / 6) * epsilon,
            -(np.sqrt(3) / 6) * epsilon,
        ]
    else:
        raise ValueError("Not valid case, must be 1, 2, or 3")

    means = list(zip(mu1, mu2))
    sims = [np.random.multivariate_normal(mean, sigma, n) for mean in means]
    if old_case == 4:
        sims[-1] = (1 - weight) * sims[-1] + weight * np.random.multivariate_normal(
            means[-1], sigma * 1.5, n
        )
    elif old_case == 5:
        sims = [
            (1 - weight) * sims[i]
            + weight * np.random.multivariate_normal(means[i], sigma * 1.5, n)
            for i in range(len(sims))
        ]

    return sims


KSAMP_SIMS = {
    "rot_ksamp": rot_ksamp,
    "gaussian_3samp": gaussian_3samp,
}


def ksamp_sim(ksim, n, **kwargs):
    r"""
    `K`-sample simulation generator.

    Takes a simulation and the required parameters, and outputs the simulated
    data matrices.

    Parameters
    ----------
    sim : str
        The name of the simulation (from the :mod:`hyppo.tools module).
    n : int
        The number of samples desired by the simulation (>= 3).
    **kwargs
        Additional keyword arguements for the desired simulation.

    Returns
    -------
    x,y : ndarray
        Simulated data matrices.
    """
    if ksim not in KSAMP_SIMS.keys():
        raise ValueError(
            "sim must be one of the following: {}".format(list(KSAMP_SIMS.keys()))
        )
    else:
        ksim = KSAMP_SIMS[ksim]

    return ksim(n=n, **kwargs)
