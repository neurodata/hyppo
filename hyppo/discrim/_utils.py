import numpy as np
from sklearn.metrics import pairwise_distances

from ..tools import check_ndarray_xy, check_reps, contains_nan, convert_xy_float64


class _CheckInputs:
    """Checks inputs for discriminability tests"""

    def __init__(self, x, y, reps=None, is_dist=False, remove_isolates=True):
        self.x = x
        self.y = y
        self.reps = reps
        self.is_distance = is_dist
        self.remove_isolates = remove_isolates

    def __call__(self):
        if len(self.x) > 1:
            if self.x[0].shape[0] != self.x[1].shape[0]:
                msg = "The input matrices do not have the same number of rows."
                raise ValueError(msg)

        tmp_ = []
        for x1 in self.x:
            check_ndarray_xy(x1, self.y)
            contains_nan(x1)
            contains_nan(self.y)
            check_min_samples(x1)
            x1, self.y = convert_xy_float64(x1, self.y)
            tmp_.append(self._condition_input(x1))

        self.x = tmp_

        if self.reps:
            check_reps(self.reps)

        return self.x, self.y

    def _condition_input(self, x1):
        """Checks whether there is only one subject and removes
        isolates and calculate distance."""
        uniques, counts = np.unique(self.y, return_counts=True)

        if (counts != 1).sum() <= 1:
            msg = "You have passed a vector containing only a single unique sample id."
            raise ValueError(msg)

        if self.remove_isolates:
            idx = np.isin(self.y, uniques[counts != 1])
            self.y = self.y[idx]

            x1 = np.asarray(x1)
            if not self.is_distance:
                x1 = x1[idx]
            else:
                x1 = x1[np.ix_(idx, idx)]

        if not self.is_distance:
            x1 = pairwise_distances(x1, metric="euclidean")

        return x1


def check_min_samples(x1):
    """Check if the number of samples is at least 3"""
    nx = x1.shape[0]

    if nx <= 10:
        raise ValueError("Number of samples is too low")
