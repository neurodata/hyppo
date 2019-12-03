import warnings
import numpy as np
from .._utils import (contains_nan, check_ndarray_xy, convert_xy_float64,
                      check_reps)

class _CheckInputs:
    """Checks inputs for discriminability tests"""
    def __init__(self, X, Y, reps = None):
        self.X = X
        self.Y = Y
        self.reps = reps

    def __call__(self):
        check_ndarray_xy(self.X, self.Y)
        contains_nan(self.X)
        contains_nan(self.Y)
        self.X, self.Y = convert_xy_float64(self.X, self.Y)
        
        if self.reps:
            check_reps(self.reps)
        
        return self.X, self.Y
    

