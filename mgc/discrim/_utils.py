import warnings
import numpy as np
from .._utils import (contains_nan, check_ndarray_xy, convert_xy_float64,
                      check_reps)

class _CheckInputs:
    """Checks inputs for discriminability tests"""
    def __init__(self, x, y, reps = None):
        self.x = x
        self.y = y
        self.reps = reps

    def __call__(self):
        check_ndarray_xy(self.x, self.y)
        contains_nan(self.x)
        contains_nan(self.y)
        self._check_min_samples()
        self.X, self.Y = convert_xy_float64(self.X, self.Y)
        
        if self.reps:
            check_reps(self.reps)
        
        return self.x, self.y

    def _check_min_samples(self):
        """Check if the number of samples is at least 3"""
        nx = self.x.shape[0]

        if nx <= 10:
            raise ValueError("Number of samples is too low")
    

