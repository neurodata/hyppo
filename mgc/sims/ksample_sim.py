import numpy as np

from .indep_sim import *


_SIMS = [
    linear,
    spiral,
    exponential,
    cubic,
    joint_normal,
    step,
    quadratic,
    w_shaped,
    uncorrelated_bernoulli,
    logarithmic,
    fourth_root,
    sin_four_pi,
    sin_sixteen_pi,
    two_parabolas,
    circle,
    ellipse,
    diamond,
    multiplicative_noise,
    square,
    multimodal_independence,
]


def _normalize(x, y):
    """Normalize input data matricies."""
    x[:, 0] = x[:, 0] / np.max(np.abs(x[:, 0]))
    y[:, 0] = y[:, 0] / np.max(np.abs(y[:, 0]))
    return x, y


def _2samp_rotate(sim, x, y, p, degree=90, pow_type="samp"):
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
    if sim.__name__ in same_shape:
        rot_shape = 2 * p
    else:
        rot_shape = p + 1
    rot_mat = np.identity(rot_shape)
    if pow_type == "dim":
        if sim.__name__ not in [
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

    if sim.__name__ in [
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
    ]:
        x_rot, y_rot = np.hsplit(rot_data, 2)
    else:
        x_rot, y_rot = np.hsplit(rot_data, [-p])

    return x_rot, y_rot


def rot_2samp(sim, n, p, noise=True, degree=90, trans=0):
    """Rotated 2 sample test"""
    if sim not in _SIMS:
        raise ValueError("Not valid simulation")

    if sim.__name__ == "multimodal_independence":
        x, y = sim(n, p)
        x_rot, y_rot = sim(n, p)
    else:
        if sim.__name__ == "multiplicative_noise":
            x, y = sim(n, p)
        else:
            x, y = sim(n, p, noise=noise)
        x_rot, y_rot = _2samp_rotate(sim, x, y, p, degree=degree, pow_type="samp")
    samp1 = np.hstack([x, y])
    samp2 = np.hstack([x_rot, y_rot])

    return samp1, samp2


def trans_2samp(sim, n, p, noise=True, degree=90, trans=0.3):
    """Translated 2 sample test"""
    if sim not in _SIMS:
        raise ValueError("Not valid simulation")

    if sim.__name__ == "multimodal_independence":
        x, y = sim(n, p)
        x_trans, y_trans = sim(n, p)
    else:
        if sim.__name__ == "multiplicative_noise":
            x, y = sim(n, p)
        else:
            x, y = sim(n, p, noise=noise)
        x, y = _normalize(x, y)
        x_trans, y_trans = _2samp_rotate(sim, x, y, p, degree=degree, pow_type="dim")
        x_trans[:, 0] += trans
        y_trans[:, 0] = y_trans[:, -1]

    samp1 = np.hstack([x, y])
    samp2 = np.hstack([x_trans, y_trans])

    return samp1, samp2


def gaussian_3samp(n, epsilon=1, weight=0, case=1):
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
        sims[-1] = (1 - weight) * sims[-1] + weight * np.random.multivariate_normal(means[-1], sigma * 4, n)
    elif old_case == 5:
        sims = [(1 - weight) * sims[i] + weight * np.random.multivariate_normal(means[i], sigma * 4, n) for i in range(len(sims))]

    return sims
