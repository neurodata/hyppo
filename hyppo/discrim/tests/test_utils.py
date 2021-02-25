import numpy as np
import pytest
from numpy.testing import assert_raises

from .._utils import _CheckInputs


class TestErrorWarn:
    """Tests errors and warnings."""

    def test_error_shape(self):
        # raises error if number of samples different (n)
        x = np.arange(100).reshape(25, 4)
        y = x.reshape(10, 10)
        assert_raises(ValueError, _CheckInputs(x, y))

    def test_error_lowsamples(self):
        # raises error if samples are low (< 3)
        x = np.arange(2).reshape(-1, 1)
        y = np.arange(2).reshape(-1, 1)
        assert_raises(ValueError, _CheckInputs(x, y))

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        x = np.arange(20, dtype=float).reshape(-1, 1)
        x[0] = np.nan
        assert_raises(ValueError, _CheckInputs(x, x))

        y = np.arange(20)
        assert_raises(ValueError, _CheckInputs(x, y))

    @pytest.mark.parametrize(
        "reps", [-1, "1"]  # reps is negative  # reps is not integer
    )
    def test_error_reps(self, reps):
        # raises error if reps is negative
        x = np.arange(20).reshape(-1, 1)
        assert_raises(ValueError, _CheckInputs(x, x, reps=reps))
