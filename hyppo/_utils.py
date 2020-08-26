import warnings
from joblib import Parallel, delayed

import numpy as np
from scipy.stats.distributions import chi2
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

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


class _PermNode(object):
    """
    Helper class for nodes in _PermTree.
    """
    def __init__(self, parent, label=None, index=None):
        self.children = []
        self.parent = parent
        self.label = label
        self.index = index

    def get_leaf_indices(self):
        if len(self.children) == 0:
            return [self.index]
        else:
            indices = []
            for child in self.children:
                indices += child.get_leaf_indices()
            return indices

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

class _PermTree(object):
    """
    Tree representation of dependencies for restricted permutations
    """
    def __init__(self, perm_blocks):
        perm_blocks = np.asarray(perm_blocks)
        if perm_blocks.ndim == 1:
            perm_blocks = perm_blocks[:,np.newaxis]
        self.root = _PermNode(None)
        self._add_levels(self.root, perm_blocks, np.arange(perm_blocks.shape[0]))
        indices = self.root.get_leaf_indices()
        self._index_order = np.argsort(indices)

    def _add_levels(self, root: _PermNode, perm_blocks, indices):
        # Add new child node for each unique label, then recurse or end
        if perm_blocks.shape[1] == 0:
            for idx in indices:
                child_node = _PermNode(parent=root, label=1, index=idx)
                root.add_child(child_node)
        else:
            for label in np.unique(perm_blocks[:, 0]):
                idxs = np.where(perm_blocks[:, 0] == label)[0]
                child_node = _PermNode(parent=root, label=label)
                root.add_child(child_node)
                self._add_levels(child_node, perm_blocks[idxs, 1:], indices[idxs])

    def _permute_level(self, node):
        if len(node.get_children()) == 0:
            return [node.index]
        else:
            indices, labels = zip(*[(self._permute_level(child), child.label) for child in node.get_children()])
            shuffle_children = [i for i,label in enumerate(labels) if label >= 0]
            indices = np.asarray(indices)
            if len(shuffle_children) > 1:
                indices[shuffle_children] = indices[np.random.permutation(shuffle_children)]
            return np.concatenate(indices)

    def permute_indices(self):
        return self._permute_level(self.root)[self._index_order]

    def original_indices(self):
        return np.arange(len(self._index_order))

# permutation group shuffling class
class _PermGroups(object):
    """
    Helper function to calculate parallel p-value.
    """
    def __init__(self, y, perm_blocks=None):
        if perm_blocks is None:
            self.perm_tree = None
        else:
            self.perm_tree = _PermTree(perm_blocks)

    def __call__(self):
        if self.perm_tree is None:
            order = np.random.permutation(self.y_labels.shape[0])
        else:
            order = self.perm_tree.permute_indices()

        return order

# p-value computation
def _perm_stat(calc_stat, x, y, is_distsim=True, permuter=None):
    if permuter is not None:
        order = permuter()
    else:
        order = np.random.permutation(y.shape[0])

    if is_distsim:
        permy = y[order][:, order]
    else:
        permy = y[order]
    perm_stat = calc_stat(x, permy)

    return perm_stat


def perm_test(calc_stat, x, y, reps=1000, workers=1, is_distsim=True, perm_blocks=None):
    """
    Calculate the p-value via permutation
    """
    # calculate observed test statistic
    stat = calc_stat(x, y)

    # calculate null distribution
    if is_distsim:
        permuter = _PermGroups(y, perm_blocks)
    else:
        permuter = None
    null_dist = np.array(
        Parallel(n_jobs=workers)(
            [delayed(_perm_stat)(calc_stat, x, y, is_distsim, permuter) for rep in range(reps)]
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
