import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_warns, assert_raises
from .. import oneSample

class TestOneSample:
    def test_same_one(self):
        x = np.ones((100,2),dtype=float)
        y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)

        np.random.seed(123456789)
        obs_stat = 0.5
        obs_p = 1
        stat, p = oneSample().test(x,y) 
        
        assert_almost_equal(stat, obs_stat, decimal=2)
        assert_almost_equal(p, obs_p, decimal=2)

    def test_diff_one(self):
        x = np.concatenate((np.zeros((50,2)) ,np.ones((50,2))), axis=0)
        y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)

        np.random.seed(123456789)
        obs_stat = 1.0
        obs_p = 0.001
        oneSamp = oneSample()
        stat, p = oneSamp.test(x,y) 

        assert_almost_equal(stat, obs_stat, decimal=3)
        assert_almost_equal(p, obs_p, decimal=3)


class TestOneSampleWarn:
    """ Tests errors and warnings derived from one sample test.
    """

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        x = np.arange(20, dtype=float)
        x[0] = np.nan
        assert_raises(ValueError, oneSample().test, x, x)

        y = np.arange(20)
        assert_raises(ValueError, oneSample().test, x, y)

    @pytest.mark.parametrize("reps", [
        -1,    # reps is negative
        '1',   # reps is not integer
    ])
    def test_error_reps(self, reps):
        # raises error if reps is negative
        x = np.ones((100,2),dtype=float)
        y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)

        assert_raises(ValueError, oneSample().test, x, y, reps=reps)

    def test_warns_reps(self):
        # raises warning when reps is less than 1000
        x = np.ones((100,2),dtype=float)
        y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)

        reps = 100
        assert_warns(RuntimeWarning, oneSample().test, x, y, reps=reps)
