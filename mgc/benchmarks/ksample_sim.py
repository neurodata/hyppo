import numpy as np

from .indep_sim import linear


__all__ = ["linear_2samp"]


def _k_sample_rotate2d(x, y, degree=45):
    angle = np.radians(degree)
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    data = np.hstack([x, y])
    rot_data = (rot_mat @ data.T).T
    x_rot, y_rot = np.hsplit(rot_data, 2)

    return x_rot, y_rot


def linear_2samp(n, p, noise=1, low=-1, high=1):
    x, y = linear(n, 1, noise=noise, low=low, high=high)
    x_rot, y_rot = _k_sample_rotate2d(x, y)
    samp1 = np.hstack([x, y])
    samp2 = np.hstack([x_rot, y_rot])

    return samp1, samp2
