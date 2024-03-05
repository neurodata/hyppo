from itertools import permutations

import numpy as np
import pytest
from numpy.testing import assert_raises

from .._utils import _CheckInputs


class TestErrorWarn:
    """Tests errors and warnings."""

    def test_error_notndarray(self):
        # raises error if x or y is not a ndarray
        x = np.arange(20)
        y = np.arange(20)
        z = [5] * 20

        for data in permutations((x, y, z)):
            assert_raises(TypeError, _CheckInputs(*data))

    def test_error_shape(self):
        # raises error if number of samples different (n)
        x = np.arange(100).reshape(25, 4)
        y = x.reshape(10, 10)
        z = x.reshape(20, 5)
        assert_raises(ValueError, _CheckInputs(x, y, z))

    def test_error_lowsamples(self):
        # raises error if samples are low (< 3)
        x = np.arange(3)
        y = np.arange(3)
        z = np.arange(3)
        assert_raises(ValueError, _CheckInputs(x, y, z))

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        x = np.arange(20, dtype=float)
        x[0] = np.nan
        assert_raises(ValueError, _CheckInputs(x, x, x))

        y = np.arange(20)
        assert_raises(ValueError, _CheckInputs(x, y, y))

    @pytest.mark.parametrize(
        "reps", [-1, "1"]  # reps is negative  # reps is not integer
    )
    def test_error_reps(self, reps):
        # raises error if reps is negative
        x = np.arange(20)
        assert_raises(ValueError, _CheckInputs(x, x, x, reps=reps))

    def test_max_sims(self):
        # raises error if data contains more dimentions than allowed
        x = np.arange(20).reshape(10, 2)
        y = np.arange(10).reshape(10, 1)
        z = y

        assert_raises(ValueError, _CheckInputs(x, y, z, max_dims=1))

    def test_constant_input(self):
        # raises error if z is constant
        x = np.arange(20).reshape(-1, 1)
        y = np.arange(20).reshape(-1, 1)
        z = np.ones(20).reshape(-1, 1)

        assert_raises(ValueError, _CheckInputs(x, y, z, ignore_z_var=False))

        try:
            _CheckInputs(x, y, z, ignore_z_var=True)
        except Exception as exc:
            assert False, f"'_CheckInputs' with 'ignore_z_var=True' with constant z input raised an exception {exc}"