import numpy as np

from .._utils import contains_nan
from ..independence import CCA, Dcorr, HHG, RV, Hsic, MGC


class _CheckInputs:
    def __init__(self, inputs, indep_test, reps=None, compute_distance=None):
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
        self._check_variance()

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
                i = i[:, np.newaxis]
            elif i.ndim != 2:
                raise ValueError(
                    "Expected a 2-D array `i`, found shape " "{}".format(i.shape)
                )
            dims.append(i.shape[1])
            new_inputs.append(i)

        self._check_nd_ksampletest(dims)

        return new_inputs

    def _check_nd_ksampletest(self, dims):
        if len(set(dims)) > 1:
            raise ValueError(
                "Shape mismatch, inputs must have shape " "[n, p] and [m, p]."
            )

    def _convert_inputs_float64(self):
        return [np.asarray(i).astype(np.float64) for i in self.inputs]

    def _check_indep_test(self):
        tests = [CCA, Dcorr, HHG, RV, Hsic, MGC]
        if self.indep_test.__class__ not in tests and self.indep_test is not None:
            raise ValueError(
                "indep_test must be CannCorr, Dcorr, HHG, RVCorr, Hsic, MGC, or MGCRF"
            )

    def _check_min_samples(self):
        for i in self.inputs:
            if i.shape[0] <= 3:
                raise ValueError("Number of samples is too low")

    def _check_variance(self):
        for i in self.inputs:
            if np.var(i) == 0:
                raise ValueError("Test cannot be run, one of the inputs has 0 variance")


def k_sample_transform(inputs):
    n_inputs = len(inputs)
    u = np.vstack(inputs)

    if n_inputs == 2:
        n1 = inputs[0].shape[0]
        n2 = inputs[1].shape[0]
        v = np.vstack([np.zeros((n1, 1)), np.ones((n2, 1))])
    else:
        vs = []
        for i in range(n_inputs):
            n = inputs[i].shape[0]
            encode = np.zeros(shape=(n, n_inputs))
            encode[:, i] = np.ones(shape=n)
            vs.append(encode)
        v = np.concatenate(vs)

    return u, v
