import numpy as np

from ..tools import check_ndarray_xyz, check_reps, contains_nan, convert_xyz_float64


class _CheckInputs:
    """Checks inputs for all independence tests"""

    def __init__(self, x, y, z, reps=None, max_dims=None, ignore_z_var=False):
        self.x = x
        self.y = y
        self.z = z
        self.reps = reps
        self.max_dims = max_dims
        self.ignore_z_var = ignore_z_var # to allow for constant z input

    def __call__(self):
        check_ndarray_xyz(self.x, self.y, self.z)
        contains_nan(self.x)
        contains_nan(self.y)
        contains_nan(self.z)
        self.x, self.y, self.z = self.check_dim_xyz(max_dims=self.max_dims)
        self.x, self.y, self.z = convert_xyz_float64(self.x, self.y, self.z)
        self._check_min_samples()
        self._check_variance()

        if self.reps:
            check_reps(self.reps)

        return self.x, self.y, self.z

    def check_dim_xyz(self, max_dims):
        """Check and convert x and y to proper dimensions"""
        if self.x.ndim == 1:
            self.x = self.x[:, np.newaxis]
        elif self.x.ndim != 2:
            raise ValueError(
                "Expected a 2-D array `x`, found shape " "{}".format(self.x.shape)
            )
        if self.y.ndim == 1:
            self.y = self.y[:, np.newaxis]
        elif self.y.ndim != 2:
            raise ValueError(
                "Expected a 2-D array `y`, found shape " "{}".format(self.y.shape)
            )
        if self.z.ndim == 1:
            self.z = self.z[:, np.newaxis]
        elif self.z.ndim != 2:
            raise ValueError(
                "Expected a 2-D array `z`, found shape " "{}".format(self.z.shape)
            )

        if max_dims is not None:
            _, dx = self.x.shape
            _, dy = self.y.shape
            _, dz = self.z.shape

            if np.any(np.array([dx, dy, dz]) > max_dims):
                raise ValueError(
                    f"x, y, z must have be univariate and have shape [n,{max_dims}]"
                )

        self._check_nd_indeptest()

        return self.x, self.y, self.z

    def _check_nd_indeptest(self):
        """Check if number of samples is the same"""
        nx, _ = self.x.shape
        ny, _ = self.y.shape
        nz, _ = self.z.shape
        if not np.all(np.array([nx, ny, nz]) == nx):
            raise ValueError(
                "Shape mismatch, x, y and z must have shape "
                + "[n, p], [n, q] and [n, r]."
            )

    def _check_min_samples(self):
        """Check if the number of samples is at least 3"""
        nx = self.x.shape[0]
        ny = self.y.shape[0]
        nz = self.z.shape[0]

        if nx <= 3 or ny <= 3 or nz <= 3:
            raise ValueError("Number of samples is too low")

    def _check_variance(self):
        if np.var(self.x) == 0:
        # or np.var(self.y) == 0 or np.var(self.z) == 0:
            raise ValueError("Test cannot be run. Input array x has 0 variance.")
        if np.var(self.y) == 0:
            raise ValueError("Test cannot be run. Input array y has 0 variance")
        if not self.ignore_z_var:
            if np.var(self.z) == 0:
                raise ValueError("Test cannot be run. Input array z has 0 variance")
