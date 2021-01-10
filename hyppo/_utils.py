import warnings
from joblib import Parallel, delayed

import numpy as np
from scipy.stats.distributions import chi2
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels

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


def check_distmat(x, y):
    """Check if x and y are distance matrices."""
    if (
        not np.array_equal(x, x.T)
        or not np.array_equal(y, y.T)
        or not np.all((x.diagonal() == 0))
        or not np.all((y.diagonal() == 0))
    ):
        raise ValueError(
            "x and y must be distance matrices, {is_sym} symmetric and {zero_diag} zeros along the diagonal".format(
                is_sym="x is not"
                if not np.array_equal(x, x.T)
                else "y is not"
                if not np.array_equal(y, y.T)
                else "both are",
                zero_diag="x doesn't have"
                if not np.all((x.diagonal() == 0))
                else "y doesn't have"
                if not np.all((y.diagonal() == 0))
                else "both have",
            )
        )


def check_kernmat(x, y):
    """Check if x and y are similarity matrices."""
    if (
        not np.array_equal(x, x.T)
        or not np.array_equal(y, y.T)
        or not np.all((x.diagonal() == 1))
        or not np.all((y.diagonal() == 1))
    ):
        raise ValueError(
            "x and y must be distance matrices, {is_sym} symmetric and {one_diag} ones along the diagonal".format(
                is_sym="x is not"
                if not np.array_equal(x, x.T)
                else "y is not"
                if not np.array_equal(y, y.T)
                else "both are",
                one_diag="x doesn't have"
                if not np.all((x.diagonal() == 1))
                else "y doesn't have"
                if not np.all((y.diagonal() == 1))
                else "both have",
            )
        )


def compute_kern(x, y, metric="gaussian", workers=None, **kwargs):
    if metric == None:
        metric = "precomputed"
    if metric == "gaussian":
        if "gamma" not in kwargs:
            l1 = pairwise_distances(x, metric="l1", n_jobs=workers)
            n = l1.shape[0]
            med = np.median(
                np.lib.stride_tricks.as_strided(
                    l1, (n - 1, n + 1), (l1.itemsize * (n + 1), l1.itemsize)
                )[:, 1:]
            )
            kwargs["gamma"] = 1.0 / (2 * (med ** 2))
        metric = "rbf"
    if callable(metric):
        simx = metric(x, **kwargs)
        simy = metric(y, **kwargs)
        check_kernmat(simx, simy)
    else:
        simx = pairwise_kernels(x, metric=metric, n_jobs=workers, **kwargs)
        simy = pairwise_kernels(y, metric=metric, n_jobs=workers, **kwargs)
    return simx, simy


def compute_dist(x, y, metric="euclidean", workers=None, **kwargs):
    if metric == None:
        metric = "precomputed"
    if callable(metric):
        distx = metric(x, **kwargs)
        disty = metric(y, **kwargs)
        check_distmat(distx, disty)
    else:
        distx = pairwise_distances(x, metric=metric, n_jobs=workers, **kwargs)
        disty = pairwise_distances(y, metric=metric, n_jobs=workers, **kwargs)
    return distx, disty


def check_perm_blocks(perm_blocks):
    # Checks generic properties of perm_blocks
    if perm_blocks is None:
        return None
    elif isinstance(perm_blocks, list):
        perm_blocks = np.asarray(perm_blocks)
    elif not isinstance(perm_blocks, np.ndarray):
        raise TypeError("perm_blocks must be an ndarray or list")
    if perm_blocks.ndim == 1:
        perm_blocks = perm_blocks[:, np.newaxis]
    elif perm_blocks.ndim > 2:
        raise ValueError("perm_blocks must be of at most dimension 2")

    return perm_blocks


def check_perm_blocks_dim(perm_blocks, y):
    if not perm_blocks.shape[0] == y.shape[0]:
        raise ValueError(f"perm_bocks first dimension must be same length as y")


def check_perm_block(perm_block):
    # checks a hierarchy level of perm_blocks for proper exchangeability
    if not isinstance(perm_block[0], int):
        unique, perm_blocks, counts = np.unique(
            perm_block, return_counts=True, return_inverse=True
        )
        pos_counts = counts
    else:
        unique, counts = np.unique(perm_block, return_counts=True)
        pos_counts = [c for c, u in zip(counts, unique) if u >= 0]
    if len(set(pos_counts)) > 1:
        raise ValueError(
            f"Exchangeable hiearchy has groups with {min(pos_counts)} to \
                {max(pos_counts)} elements"
        )

    return perm_block


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
        perm_blocks = check_perm_blocks(perm_blocks)
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
            perm_block = check_perm_block(perm_blocks[:, 0])
            for label in np.unique(perm_block):
                idxs = np.where(perm_block == label)[0]
                child_node = _PermNode(parent=root, label=label)
                root.add_child(child_node)
                self._add_levels(child_node, perm_blocks[idxs, 1:], indices[idxs])

    def _permute_level(self, node):
        if len(node.get_children()) == 0:
            return [node.index]
        else:
            indices, labels = zip(
                *[
                    (self._permute_level(child), child.label)
                    for child in node.get_children()
                ]
            )
            shuffle_children = [i for i, label in enumerate(labels) if label >= 0]
            indices = np.asarray(indices)
            if len(shuffle_children) > 1:
                indices[shuffle_children] = indices[
                    np.random.permutation(shuffle_children)
                ]
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
        self.n = y.shape[0]
        if perm_blocks is None:
            self.perm_tree = None
        else:
            self.perm_tree = _PermTree(perm_blocks)

    def __call__(self):
        if self.perm_tree is None:
            order = np.random.permutation(self.n)
        else:
            order = self.perm_tree.permute_indices()

        return order


# p-value computation
def _perm_stat(calc_stat, x, y, is_distsim=True, permuter=None):
    if permuter is None:
        order = np.random.permutation(y.shape[0])
    else:
        order = permuter()

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
    permuter = _PermGroups(y, perm_blocks)
    null_dist = np.array(
        Parallel(n_jobs=workers)(
            [
                delayed(_perm_stat)(calc_stat, x, y, is_distsim, permuter)
                for rep in range(reps)
            ]
        )
    )
    pvalue = (1 + (null_dist >= stat).sum()) / (1 + reps)

    return stat, pvalue, null_dist


def chi2_approx(calc_stat, x, y):
    n = x.shape[0]
    stat = calc_stat(x, y)
    pvalue = chi2.sf(stat * n + 1, 1)

    return stat, pvalue
