import warnings

import numpy as np

from .._utils import (contains_nan, check_ndarray_inputs,
                      convert_inputs_float64, check_reps,
                      check_compute_distance)
from ..independence import *


class _CheckInputs:
    def __init__(self, inputs, dim, indep_test, reps=None,
                 compute_distance=None):
        self.inputs = inputs
        self.dim = dim
        self.compute_distance = compute_distance
        self.reps = reps
        self.indep_test = indep_test

    def __call__(self, test_name):
        check_ndarray_inputs(self.inputs)
        for i in self.inputs:
            contains_nan(i)
        self.inputs = self.check_dim(test_name)
        self.inputs = convert_inputs_float64(self.inputs)
        self._check_indep_test()
        self._check_min_samples()
        check_compute_distance(self.compute_distance)

        if self.reps:
            check_reps(self.reps)

        return self.inputs

    def check_dim(self, test_name):
        # check if inputs are ndarrays
        new_inputs = []
        dims = []
        for i in self.inputs:
            if self.dim == 1:
                # check if x or y is shape (n,)
                if i.ndim > 1:
                    msg = ("inputs must be of shape (n,). Will reshape")
                    warnings.warn(msg, RuntimeWarning)
                    i.shape = (-1)
            elif self.dim > 1:
                # convert arrays of type (n,) to (n, 1)
                if i.ndim == 1:
                    i.shape = (-1, 1)
                dims.append(i.shape[1])
            new_inputs.append(i)

        if self.dim > 1:
            self._check_nd_ksampletest(dims, test_name)

        return new_inputs

    def _check_nd_ksampletest(self, dims, test_name):
        test_psame = ["UnpairKSample"]
        if test_name in test_psame:
            if len(set(dims)) > 1:
                raise ValueError("Shape mismatch, inputs must have shape "
                                 "[n, p] and [m, p].")

    def _check_indep_test(self):
        tests = [CannCorr, Dcorr, HHG, RVCorr]
        if self.indep_test.__class__ not in tests:
            raise ValueError("Independence test must be implemented in mgc")

    def _check_min_samples(self):
        for i in self.inputs:
            if i.shape[0] <= 3:
                raise ValueError("Number of samples is too low")


def k_sample_transform(inputs):
    n_inputs = len(inputs)
    u = np.vstack(inputs)
    ns = [i.shape[0] for i in inputs]
    v_list = [np.repeat(i/(n_inputs-i), ns[i]) for i in range(n_inputs)]
    v = np.vstack(v_list).reshape(-1, 1)

    return u, v