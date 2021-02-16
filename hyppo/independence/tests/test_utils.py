import numpy as np
import pytest
from numpy.testing import assert_raises
from sklearn.ensemble import RandomForestRegressor

from ...tools.common import _check_kernmat
from .._utils import _CheckInputs, sim_matrix


class TestErrorWarn:
    """Tests errors and warnings."""

    def test_error_notndarray(self):
        # raises error if x or y is not a ndarray
        x = np.arange(20)
        y = [5] * 20
        assert_raises(TypeError, _CheckInputs(x, y))
        assert_raises(TypeError, _CheckInputs(y, x))

    def test_error_shape(self):
        # raises error if number of samples different (n)
        x = np.arange(100).reshape(25, 4)
        y = x.reshape(10, 10)
        assert_raises(ValueError, _CheckInputs(x, y))

    def test_error_lowsamples(self):
        # raises error if samples are low (< 3)
        x = np.arange(3)
        y = np.arange(3)
        assert_raises(ValueError, _CheckInputs(x, y))

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        x = np.arange(20, dtype=float)
        x[0] = np.nan
        assert_raises(ValueError, _CheckInputs(x, x))

        y = np.arange(20)
        assert_raises(ValueError, _CheckInputs(x, y))

    @pytest.mark.parametrize(
        "reps", [-1, "1"]  # reps is negative  # reps is not integer
    )
    def test_error_reps(self, reps):
        # raises error if reps is negative
        x = np.arange(20)
        assert_raises(ValueError, _CheckInputs(x, x, reps=reps))


class TestHelper:
    def test_simmat(self):
        # raises error if x or y is not a ndarray
        clf = RandomForestRegressor()
        x = np.arange(20).reshape(-1, 1)
        y = np.arange(5, 25)
        clf.fit(x, y)
        kernx = sim_matrix(clf, x)
        _check_kernmat(kernx, kernx)
