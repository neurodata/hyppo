import warnings

import numpy as np

from .._utils import contains_nan
from ..independence import CCA, Dcorr, HHG, RV, Hsic


class _CheckInputs:
    def __init__(self, inputs, indep_test, reps=None,
                 compute_distance=None):
        self.inputs = inputs
        self.compute_distance = compute_distance
        self.reps = reps
        self.indep_test = indep_test

    def __call__(self):
        self._check_ndarray_inputs()
        for i in self.inputs:
            contains_nan(i)
        self.inputs = self.check_dim()
        self.inputs = self._convert_inputs_float64()
        self._check_indep_test()
        self._check_min_samples()

        return self.inputs

    def _check_ndarray_inputs(self):
        if len(self.inputs) < 2:
            raise ValueError("there must be at least 2 inputs")
        for i in self.inputs:
            if not isinstance(i, np.ndarray):
                raise ValueError("x and y must be ndarrays")

    def check_dim(self):
        # check if inputs are ndarrays
        new_inputs = []
        dims = []
        for i in self.inputs:
            # convert arrays of type (n,) to (n, 1)
            if i.ndim == 1:
                i.shape = (-1, 1)
            dims.append(i.shape[1])
            new_inputs.append(i)

        self._check_nd_ksampletest(dims)

        return new_inputs

    def _check_nd_ksampletest(self, dims):
        if len(set(dims)) > 1:
            raise ValueError("Shape mismatch, inputs must have shape "
                                "[n, p] and [m, p].")

    def _convert_inputs_float64(self):
        return [np.asarray(i).astype(np.float64) for i in self.inputs]

    def _check_indep_test(self):
        tests = [CCA, Dcorr, HHG, RV, Hsic]
        if self.indep_test.__class__ not in tests:
            raise ValueError("indep_test must be CannCorr, Dcorr, HHG, "
                             "RVCorr, or Hsic")

    def _check_min_samples(self):
        for i in self.inputs:
            if i.shape[0] <= 3:
                raise ValueError("Number of samples is too low")


def k_sample_transform(inputs):
    n_inputs = len(inputs)
    u = np.vstack(inputs)
    ns = [i.shape[0] for i in inputs]
    v = np.concatenate([np.repeat(i, ns[i]) for i in range(n_inputs)]).reshape(-1, 1)

    return u, v