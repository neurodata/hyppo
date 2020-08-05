import warnings
from joblib import Parallel, delayed

import numpy as np
from scipy.stats.distributions import chi2
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from collections import defaultdict
from copy import deepcopy

# from scipy
def contains_nan(a):
    """Check if inputs contains NaNs"""
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid="ignore"):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = np.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            msg = (
                "The input array could not be properly checked for nan "
                "values. nan values will be ignored."
            )
            warnings.warn(msg, RuntimeWarning)

    if contains_nan:
        raise ValueError("Input contains NaNs. Please omit and try again")

    return contains_nan


def check_ndarray_xy(x, y):
    """Check if x or y is an ndarray"""
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("x and y must be ndarrays")


def convert_xy_float64(x, y):
    """Convert x or y to np.float64 (if not already done)"""
    # convert x and y to floats
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)

    return x, y


def check_reps(reps):
    """Check if reps is valid"""
    # check if reps is an integer > than 0
    if not isinstance(reps, int) or reps < 0:
        raise ValueError("Number of reps must be an integer greater than 0.")

    # check if reps is under 1000 (recommended)
    elif reps < 1000:
        msg = (
            "The number of replications is low (under 1000), and p-value "
            "calculations may be unreliable. Use the p-value result, with "
            "caution!"
        )
        warnings.warn(msg, RuntimeWarning)


def check_compute_distance(compute):
    """Check if compute distance/kernel function if a callable()"""
    if not callable(compute) and compute is not None:
        raise ValueError("The compute distance/kernel must be a function.")


def check_xy_distmat(x, y):
    """Check if x and y are distance matrices"""
    nx, px = x.shape
    ny, py = y.shape
    if nx != px or ny != py or np.trace(x) != 0 or np.trace(y) != 0:
        raise ValueError(
            "Shape mismatch, x and y must be distance matrices "
            "have shape [n, n] and [n, n]."
        )


def check_inputs_distmat(inputs):
    # check if x and y are distance matrices
    for i in inputs:
        n, p = i.shape
        if n != p or np.trace(i) != 0:
            raise ValueError(
                "Shape mismatch, x and y must be distance matrices "
                "have shape [n, n] and [n, n]."
            )


def euclidean(x, workers=None):
    """Default euclidean distance function calculation"""
    return pairwise_distances(X=x, metric="euclidean", n_jobs=workers)


def gaussian(x, workers=None):
    """Default medial gaussian kernel similarity calculation"""
    l1 = pairwise_distances(X=x, metric="l1", n_jobs=workers)
    n = l1.shape[0]
    med = np.median(
        np.lib.stride_tricks.as_strided(
            l1, (n - 1, n + 1), (l1.itemsize * (n + 1), l1.itemsize)
        )[:, 1:]
    )
    gamma = 1.0 / (2 * (med ** 2))
    return rbf_kernel(x, gamma=gamma)

# permutation group shuffling class
class _PermGroups(object):
    """
    Helper function to calculate parallel p-value.
    """
    def __init__(self, y, permute_groups=None, permute_structure=None):
        self.permute_groups = permute_groups
        self.permute_structure = permute_structure
        self.y_labels = np.unique(y, return_inverse=True, axis=1)[1]
        if permute_structure == 'within':
            self.group_indices = defaultdict(list)
            for i,group in enumerate(permute_groups):
                self.group_indices[group].append(i)
        elif permute_structure == 'across':
            # dict: [y_label] -> list(indices)
            self.class_indices = defaultdict(list) 
            group_indices = defaultdict(list)
            for i,(group,label) in enumerate(zip(permute_groups, self.y_labels)):
                self.class_indices[label].append(i)
                group_indices[group].append(i)
            # list of group indices, sorted descending order
            self.group_indices = sorted(
                group_indices.values(), key=lambda x: len(x), reverse=True
            )

    def __call__(self):
        if self.permute_groups is None or self.permute_structure=='full':
            order = np.random.permutation(self.y_labels.shape[0])
        elif self.permute_structure == 'within':
            old_indices = np.hstack(list(self.group_indices.values()))
            new_indices = np.hstack([np.random.permutation(idx) for idx in self.group_indices.values()])
            order = np.ones(self.y_labels.shape[0]) * -1
            order[np.asarray(old_indices)] = new_indices
            order = order.astype(int)
        elif self.permute_structure == 'across':
            # Copy dict: [y_label] -> list(indices)
            class_indices_copy = deepcopy(self.class_indices)
            new_indices = []
            old_indices = []
            for group in self.group_indices:
                p0 = self.factorial(len(class_indices_copy[0]), len(group))
                p1 = self.factorial(len(class_indices_copy[1]), len(group))
                # New indices sampled per probabilities at that step
                if np.random.uniform() < p0 / (p0+p1):
                    new_indices += [class_indices_copy[0].pop() for _ in range(len(group))]
                else:
                    new_indices += [class_indices_copy[1].pop() for _ in range(len(group))]
                # Old indices in correct order
                old_indices += group
            order = np.ones(self.y_labels.shape[0]) * -1
            order[np.asarray(old_indices)] = new_indices
            order = order.astype(int)
        else:
            msg = "permute_structure must be of {'full', 'within', 'across'}"
            raise ValueError(msg)

        return order

    def factorial(self, n, n_mults):
        if n_mults == 0:
            return 1
        else:
            return n * self.factorial(n-1, n_mults-1)


# p-value computation
def _perm_stat(calc_stat, x, y, is_distsim=True, permuter=None, permute_groups=None):
    if is_distsim:
        order = permuter()
        permy = y[order][:, order]
    else:
        permy = np.random.permutation(y)
    perm_stat = calc_stat(x, permy, permute_groups)

    return perm_stat


def perm_test(calc_stat, x, y, reps=1000, workers=1, is_distsim=True, permute_groups=None, permute_structure=None):
    """
    Calculate the p-value via permutation
    """
    # calculate observed test statistic
    stat = calc_stat(x, y, permute_groups)

    # calculate null distribution
    if is_distsim:
        permuter = _PermGroups(y, permute_groups, permute_structure)
    else:
        permuter = None
    null_dist = np.array(
        Parallel(n_jobs=workers)(
            [delayed(_perm_stat)(calc_stat, x, y, is_distsim, permuter, permute_groups) for rep in range(reps)]
        )
    )
    pvalue = (null_dist >= stat).sum() / reps

    # correct for a p-value of 0. This is because, with bootstrapping
    # permutations, a p-value of 0 is incorrect
    if pvalue == 0:
        pvalue = 1 / reps

    return stat, pvalue, null_dist


def chi2_approx(calc_stat, x, y):
    n = x.shape[0]
    stat = calc_stat(x, y)
    pvalue = chi2.sf(stat * n + 1, 1)

    return stat, pvalue
