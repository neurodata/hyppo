import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_warns, assert_raises
from .. import DiscrimTwoSample

class TestTwoSample:
    def test_greater(self):
        X1 = np.ones((100,2),dtype=float)
        X2 = np.concatenate((np.zeros((50,2)),np.ones((50,2))), axis= 0)
        Y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)

        np.random.seed(123456789)
        obs_D1 = 0.5
        obs_D2 = 1.0
        obs_p = 1.0
        D1, D2, p = DiscrimTwoSample().test(X1,X2,Y) 
        
        assert_almost_equal(D1, obs_D1, decimal=2)
        assert_almost_equal(D2, obs_D2, decimal=2)
        assert_almost_equal(p, obs_p, decimal=2)

    def test_lesser(self):
        X1 = np.ones((100,2),dtype=float)
        X2 = np.concatenate((np.zeros((50,2)),np.ones((50,2))), axis= 0)
        Y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)

        np.random.seed(123456789)
        obs_D1 = 0.5
        obs_D2 = 1.0
        obs_p = 0.000999000999000999
        D1, D2, p = DiscrimTwoSample().test(X1,X2,Y,alt="less") 
        
        assert_almost_equal(D1, obs_D1, decimal=2)
        assert_almost_equal(D2, obs_D2, decimal=2)
        assert_almost_equal(p, obs_p, decimal=4)

    def test_neq(self):
        X1 = np.ones((100,2),dtype=float)
        X2 = np.concatenate((np.zeros((50,2)),np.ones((50,2))), axis= 0)
        Y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)

        np.random.seed(123456789)
        obs_D1 = 0.5
        obs_D2 = 1.0
        obs_p = 0.000999000999000999
        D1, D2, p = DiscrimTwoSample().test(X1,X2,Y,alt="neq") 
        
        assert_almost_equal(D1, obs_D1, decimal=2)
        assert_almost_equal(D2, obs_D2, decimal=2)
        assert_almost_equal(p, obs_p, decimal=4)


class TestTwoSampleWarn:
    """ 
    Tests errors and warnings derived from one sample test.
    """
    def test_error_ndarray(self):
        X1 = list(np.ones((100,2),dtype=float))
        X2 = list(np.concatenate((np.zeros((50,2)),np.ones((50,2))), axis= 0))
        Y = list(np.concatenate((np.zeros(50),np.ones(50)), axis= 0))

        assert_raises(ValueError, DiscrimTwoSample().test, X1, X2, Y)

    def test_error_one_id(self):
        X1 = np.ones((100,2),dtype=float)
        X2 = np.concatenate((np.zeros((50,2)),np.ones((50,2))), axis= 0)
        Y = np.ones((100,1))

        assert_raises(ValueError, DiscrimTwoSample().test, X1, X2, Y)

    def test_error_inequal_row(self):
        X1 = np.ones((99,2),dtype=float)
        X2 = np.concatenate((np.zeros((50,2)),np.ones((50,2))), axis= 0)
        Y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)

        assert_raises(ValueError, DiscrimTwoSample().test, X1, X2, Y)

    def test_error_nans(self):
        # raises error if inputs contain NaNs
        X1 = np.ones((100,2),dtype=float)
        X2 = np.concatenate((np.zeros((50,2)),np.ones((50,2))), axis= 0)
        Y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)

        X1[0] = np.nan
        assert_raises(ValueError, DiscrimTwoSample().test, X1, X2, Y)

        X1[0] = 1
        X2[0] = np.nan
        assert_raises(ValueError, DiscrimTwoSample().test, X1, X2, Y)

        X2[0] = 1
        Y[0] = np.nan
        assert_raises(ValueError, DiscrimTwoSample().test, X1, X2, Y)

    def test_error_alt(self):
        # raises error if inputs contain NaNs
        X1 = np.ones((100,2),dtype=float)
        X2 = np.concatenate((np.zeros((50,2)),np.ones((50,2))), axis= 0)
        Y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)

        X1[0] = np.nan
        assert_raises(ValueError, DiscrimTwoSample().test, X1, X2, Y, alt="abc")

    @pytest.mark.parametrize("reps", [
        -1,    # reps is negative
        '1',   # reps is not integer
    ])
    def test_error_reps(self, reps):
        # raises error if reps is negative
        X1 = np.ones((100,2),dtype=float)
        X2 = np.concatenate((np.zeros((50,2)),np.ones((50,2))), axis= 0)
        Y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)

        assert_raises(ValueError, DiscrimTwoSample().test, X1, X2, Y, reps=reps)

    def test_warns_reps(self):
        # raises warning when reps is less than 1000
        X1 = np.ones((100,2),dtype=float)
        X2 = np.concatenate((np.zeros((50,2)),np.ones((50,2))), axis= 0)
        Y = np.concatenate((np.zeros(50),np.ones(50)), axis= 0)

        reps = 100
        assert_warns(RuntimeWarning, DiscrimTwoSample().test, X1, X2, Y, reps=reps)
