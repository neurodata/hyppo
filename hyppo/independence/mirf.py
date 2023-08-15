from typing import NamedTuple

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sktree.ensemble import HonestForestClassifier

from ._utils import _CheckInputs
from .base import IndependenceTest


def auc_calibrator(tree, X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, indices, test_size=test_size
    )
    ## indicine of test set
    tree.fit(X_train, y_train)
    y_pred = tree.predict_proba(X_test)[:,1].reshape((X_test.shape[0]), 1)
    ### save the y_pred
    posterior_ind = np.hstack(
        (
            indices_test.reshape((len(indices_test), 1)),
            y_pred,
            y_test.reshape((X_test.shape[0]), 1),
        )
    )
    return posterior_ind


def perm_stat(clf, x, z, y, random_state=None):
    permuted_Z = np.random.permutation(z)
    X_permutedZ = np.hstack((x, permuted_Z))
    perm_stat = clf.statistic(X_permutedZ, y)
    return perm_stat


def perm_half(clf, z, y, x_pos):
    permuted_Z = np.random.permutation(z)
    perm_stat, perm_pos = clf.statistic(permuted_Z, y, return_pos=True)
    null_pos = forest_pos(x_pos + perm_pos)
    null_stat = roc_auc_score(null_pos[:, 1], null_pos[:, 2], max_fpr=clf.limit)

    return null_stat


def pos_diff(observe_pos, perm_pos, limit):
    total_pos = np.random.shuffle(np.concatenate((observe_pos, perm_pos)))

    half_ind = len(total_pos) * 0.5
    half_pos = total_pos[:half_ind]
    end_pos = total_pos[half_ind:]

    half_pos_final = forest_pos(half_pos)
    half_stat = roc_auc_score(half_pos_final[:, 1], half_pos_final[:, 2], max_fpr=limit)

    end_pos_final = forest_pos(end_pos)
    end_stat = roc_auc_score(end_pos_final[:, 1], end_pos_final[:, 2], max_fpr=limit)

    return abs(half_stat - end_stat)


def forest_pos(posterior):
    posterior_matrix = np.array(posterior).reshape(-1, 3)
    posterior_sorted = posterior_matrix[posterior_matrix[:, 0].argsort()]
    posterior_unique = np.unique(posterior_sorted[:, [0, 2]], axis=0)
    posterior_final = np.hstack(
        (posterior_unique, np.zeros((posterior_unique.shape[0], 1)))
    )

    ### get final posterior over the forest
    for i in posterior_final[:, 0]:
        posterior_final[posterior_final[:, 0] == i, 2] = np.mean(
            posterior_sorted[posterior_sorted[:, 0] == i, :][:, 1]
        )
    return posterior_final


class MIRFTestOutput(NamedTuple):
    stat: float
    pvalue: float
    # mirf_dict: dict


class MIRF(IndependenceTest):
    r"""
    Independence test using mutual information as the test statistic.

    Parameters
    ----------
    n_estimators : int, default: 100
        The number of trees in the forest.
    honest_fraction : float, default=0.5
        Fraction of training samples used for estimates in the trees. The
        remaining samples will be used to learn the tree structure. A larger
        fraction creates shallower trees with lower variance estimates.
    honest_prior : {"ignore", "uniform", "empirical"}, default="empirical"
        Method for dealing with empty leaves during evaluation of a test
        sample. If "ignore", the tree is ignored. If "uniform", the prior tree
        posterior is 1/(number of classes). If "empirical", the prior tree
        posterior is the relative class frequency in the voting subsample.
        If all trees are ignored, the empirical estimate is returned.
    **kwargs
        Additional arguments used for the forest (see
        :class:`sktree.ensemble.HonestForestClassifier`)
    """

    def __init__(
        self,
        n_estimators=500,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        honest_prior="empirical",
        honest_fraction=0.5,
        tree_estimator=None,
    ):
        self.clf = HonestForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            honest_prior=honest_prior,
            honest_fraction=honest_fraction,
            tree_estimator=tree_estimator,
        )
        IndependenceTest.__init__(self)

    def statistic(self, x, y, return_pos=False):
        r"""
        Helper function that calculates the MI test statistic.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the number of samples and `p` is the number of
            dimensions.

        Returns
        -------
        stat : float
            The computed MI statistic.
        """
        self.clf.fit(x, y.ravel())
        posterior = self.clf.predict_proba(x)
        H_YX = np.mean(entropy(posterior, base=np.exp(1), axis=1))
        _, counts = np.unique(y, return_counts=True)
        H_Y = entropy(counts, base=np.exp(1))
        stat = max(H_Y - H_YX, 0)

        self.stat = stat

        if return_pos:
            return self.stat, posterior

        return self.stat

    def test(self, x, y, reps=1000, workers=1, random_state=None):
        r"""
        Calculates the MI test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the number of samples and `p` is the number of
            dimensions.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed MI statistic.
        pvalue : float
            The computed MI p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.independence import MI
        >>> x = np.arange(100)
        >>> y = x
        >>> '%.1f, %.2f' % MI().test(x, y)[:1] # doctest: +SKIP
        '1.0, 0.001'
        """
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        stat, pvalue = super(MIRF, self).test(
            x, y, reps, workers, is_distsim=False, random_state=random_state
        )
        # mirf_dict = {}

        return MIRFTestOutput(stat, pvalue)  # , mirf_dict)


class MIRF_AUC(IndependenceTest):
    def __init__(
        self,
        n_estimators=500,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        honest_prior="empirical",
        honest_fraction=0.5,
        tree_estimator=None,
        limit=0.05,
    ):
        self.clf = HonestForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            honest_prior=honest_prior,
            honest_fraction=honest_fraction,
            tree_estimator=tree_estimator,
        )
        self.limit = limit
        IndependenceTest.__init__(self)

    def statistic(self, x, y, workers=1, test_size=0.2, initial=True, return_pos=False):
        # Initialize trees
        if initial:
            self.clf.fit(x, y.ravel())

        # Compute posteriors with train test splits
        posterior = Parallel(n_jobs=workers)(
            delayed(auc_calibrator)(tree, x, y, test_size)
            for tree in (self.clf.estimators_)
        )

        posterior_final = forest_pos(posterior)
        self.stat = roc_auc_score(
            posterior_final[:, 1], posterior_final[:, 2], max_fpr=self.limit
        )

        if return_pos:
            return self.stat, posterior

        return self.stat

    def test(self, x, y, reps=1000, workers=1, random_state=None):
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        stat, pvalue = super(MIRF, self).test(
            x, y, reps, workers, is_distsim=False, random_state=random_state
        )

        return MIRFTestOutput(stat, pvalue)


class MIRF_MV(IndependenceTest):
    def __init__(
        self,
        n_estimators=500,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        honest_prior="empirical",
        honest_fraction=0.5,
        tree_estimator=None,
        limit=0.05,
    ):
        self.clf = HonestForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            honest_prior=honest_prior,
            honest_fraction=honest_fraction,
            tree_estimator=tree_estimator,
        )
        self.limit = limit
        IndependenceTest.__init__(self)

    def statistic(self, x, y, workers=1, test_size=0.2, initial=True, return_pos=False):
        # Initialize trees
        if initial:
            self.clf.fit(x, y.ravel())

        # Compute posteriors with train test splits
        posterior = Parallel(n_jobs=workers)(
            delayed(auc_calibrator)(tree, x, y, test_size)
            for tree in (self.clf.estimators_)
        )

        posterior_final = forest_pos(posterior)
        self.stat = roc_auc_score(
            posterior_final[:, 1], posterior_final[:, 2], max_fpr=self.limit
        )

        if return_pos:
            return self.stat, posterior

        return self.stat

    def test(self, x, z, y, reps=1000, workers=1, random_state=None):
        XZ = np.hstack((x, z))
        observe_stat = self.statistic(XZ, y)

        null_dist = np.array(
            Parallel(n_jobs=workers)(
                [delayed(perm_stat)(self, x, z, y) for _ in range(reps)]
            )
        )
        pval = (1 + (null_dist >= observe_stat).sum()) / (1 + reps)

        return observe_stat, null_dist, pval

    def test_twin(self, x, z, y, reps=1000, workers=1, random_state=None):
        x_stat, x_pos = self.statistic(x, y, return_pos=True)

        # TODO: determine whether we need the forest

        z_stat, z_pos = self.statistic(z, y, return_pos=True)

        observe_pos = forest_pos(x_pos + z_pos)
        observe_stat = roc_auc_score(
            observe_pos[:, 1], observe_pos[:, 2], max_fpr=self.limit
        )

        null_dist = np.array(
            Parallel(n_jobs=workers)(
                [delayed(perm_half)(self, z, y, x_pos) for _ in range(reps)]
            )
        )
        pval = (1 + (null_dist >= observe_stat).sum()) / (1 + reps)

        return observe_stat, null_dist, pval

    def test_diff(self, x, z, y, reps=1000, workers=1):
        XZ = np.hstack((x, z))
        observe_stat, observe_pos = self.statistic(XZ, y, return_pos=True)

        # Compute statistic for permuted sets
        permuted_Z = np.random.permutation(z)
        X_permutedZ = np.hstack((x, permuted_Z))
        perm_stat, perm_pos = self.statistic(X_permutedZ, y, return_pos=True)

        # Boostrap sample the posterior from the two forests
        null_stats = np.array(
            Parallel(n_jobs=workers)(
                [
                    delayed(pos_diff)(observe_pos, perm_pos, limit=self.limit)
                    for _ in range(reps)
                ]
            )
        )

        stat = observe_stat - perm_stat

        pval = (1 + (null_stats >= stat).sum()) / (1 + reps)
        return stat, null_stats, pval
