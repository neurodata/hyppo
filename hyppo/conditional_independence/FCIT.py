import time
import joblib

import numpy as np
from scipy.stats import ttest_1samp
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor

from .base import ConditionalIndependenceTest, ConditionalIndependenceTestOutput


class FCIT(ConditionalIndependenceTest):
    r"""
    Fast Conditional Independence test statistic and p-value

    The Fast Conditional Independence Test is a non-parametric
    conditional independence test. :footcite:p:`chalupka2018FastConditionalIndependence`.

    Parameters
    ----------
    model: Sklearn regressor


    cv_grid: dict
       Dictionary of parameters to cross-validate over.

    Notes
    -----


    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        model=DecisionTreeRegressor(),
        cv_grid={"min_samples_split": [2, 8, 64, 512, 1e-2, 0.2, 0.4]},
        num_perm=8,
        prop_test=0.1,
        discrete=(False, False),
    ):

        self.model = model
        self.cv_grid = cv_grid
        self.num_perm = num_perm
        self.prop_test = prop_test
        self.discrete = discrete
        ConditionalIndependenceTest.__init__(self)

    def statistic(self, x, y, z=None):
        r"""
        Calculates the FCIT test statistic.

        Parameters
        ----------
        x,y,z : ndarray of float
            Input data matrices.

        Returns
        -------
        stat : float
            The computed FCIT statistic.
        """

        n_samples = x.shape[0]
        n_test = int(n_samples * self.prop_test)

        data_permutations = [
            np.random.permutation(x.shape[0]) for i in range(self.num_perm)
        ]

        clf = cross_val(x, y, z, self.cv_grid, self.model, prop_test=self.prop_test)
        datadict = {
            "x": x,
            "y": y,
            "z": z,
            "data_permutation": data_permutations,
            "n_test": n_test,
            "reshuffle": False,
            "clf": clf,
        }
        d1_stats = np.array(
            joblib.Parallel(n_jobs=-1, max_nbytes=100e6)(
                joblib.delayed(obtain_error)((datadict, i))
                for i in range(self.num_perm)
            )
        )

        if z.shape[1] == 0:
            x_indep_y = x[np.random.permutation(n_samples)]
        else:
            x_indep_y = np.empty([x.shape[0], 0])

        clf = cross_val(
            x_indep_y, y, z, self.cv_grid, self.model, prop_test=self.prop_test
        )

        datadict["reshuffle"] = True
        datadict["x"] = x_indep_y
        d0_stats = np.array(
            joblib.Parallel(n_jobs=-1, max_nbytes=100e6)(
                joblib.delayed(obtain_error)((datadict, i))
                for i in range(self.num_perm)
            )
        )

        t, p_value = ttest_1samp(d0_stats / d1_stats, 1)
        if t < 0:
            p_value = 1 - p_value / 2
        else:
            p_value = p_value / 2

        return t, p_value

    def test(self, x, y, z=None):

        n_samples = x.shape[0]

        if z is None:
            z = np.empty([n_samples, 0])

        if self.discrete[0] and not self.discrete[1]:
            x, y = y, x
        elif x.shape[1] < y.shape[1]:
            x, y = y, x

        y = StandardScaler().fit_transform(y)

        stat, pvalue = self.statistic(x, y, z)

        return ConditionalIndependenceTestOutput(stat, pvalue)


def cross_val(x, y, z, cv_grid, model, prop_test):
    """Choose the best decision tree hyperparameters by
    cross-validation. The hyperparameter to optimize is min_samples_split
    (see sklearn's DecisionTreeRegressor).
    Args:
        x (n_samples, x_dim): Input data array.
        y (n_samples, y_dim): Output data array.
        z (n_samples, z_dim): Optional auxiliary input data.
        cv_grid (dict): List of hyperparameter values to try in CV.
        regressor (sklearn classifier): Which regression model to use.
        prop_test (float): Proportion of validation data to use.
    Returns:
        DecisionTreeRegressor with the best hyperparameter setting.
    """

    splitter = ShuffleSplit(n_splits=3, test_size=prop_test)
    cv = GridSearchCV(estimator=model, cv=splitter, param_grid=cv_grid, n_jobs=-1)
    cv.fit(interleave(x, z), y)

    return type(model)(**cv.best_params_)


def interleave(x, z, seed=None):
    """Interleave x and z dimension-wise.
    Args:
        x (n_samples, x_dim) array.
        z (n_samples, z_dim) array.
    Returns
        An array of shape (n_samples, x_dim + z_dim) in which
            the columns of x and z are interleaved at random.
    """
    state = np.random.get_state()
    np.random.seed(seed or int(time.time()))
    total_ids = np.random.permutation(x.shape[1] + z.shape[1])
    np.random.set_state(state)
    out = np.zeros([x.shape[0], x.shape[1] + z.shape[1]])
    out[:, total_ids[: x.shape[1]]] = x
    out[:, total_ids[x.shape[1] :]] = z
    return out


def obtain_error(data_and_i):
    """
    A function used for multithreaded computation of the fcit test statistic.
    data['x']: First variable.
    data['y']: Second variable.
    data['z']: Conditioning variable.
    data['data_permutation']: Permuted indices of the data.
    data['perm_ids']: Permutation for the bootstrap.
    data['n_test']: Number of test points.
    data['clf']: Decision tree regressor.
    """
    data, i = data_and_i
    x = data["x"]
    y = data["y"]
    z = data["z"]
    if data["reshuffle"]:
        perm_ids = np.random.permutation(x.shape[0])
    else:
        perm_ids = np.arange(x.shape[0])
    data_permutation = data["data_permutation"][i]
    n_test = data["n_test"]
    clf = data["clf"]

    x_z = interleave(x[perm_ids], z, seed=i)

    clf.fit(x_z[data_permutation][n_test:], y[data_permutation][n_test:])
    return mse(
        y[data_permutation][:n_test], clf.predict(x_z[data_permutation][:n_test])
    )
