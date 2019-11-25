import warnings
import numpy as np 
from sklearn.utils import check_X_y


class _CheckInputs:
    """Checks inputs for discriminability tests"""
    def __init__(self, X, Y, reps = None):
        self.X = X
        self.Y = Y
        self.reps = reps

    def __call__(self):
        check_X_y(self.X, self.Y, accept_sparse=True)
        self.check_reps()

        return self.X, self.Y

    def check_reps(self):
        """Check if reps is valid"""

        # check if reps is an integer > than 0
        if not isinstance(self.reps, int) or self.reps < 0:
           raise ValueError("Number of reps must be an integer greater than 0.")

        # check if reps is under 1000 (recommended)
        elif self.reps < 1000:
            msg = ("The number of replications is low (under 1000), and p-value "
                    "calculations may be unreliable. Use the p-value result, with "
                    "caution!")
            warnings.warn(msg, RuntimeWarning)
