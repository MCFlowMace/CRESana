

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

import numpy as np

class Sampler:

    def __init__(self, sr, N):

        self._sr = sr
        self._dt = 1/sr
        self._N = N

    def _get_sample_times(self, N, t_0):

        return np.arange(N)*self._dt + t_0

    def __call__(self, N=None, t_0=0):

        n_samples = N
        if n_samples==None:
            n_samples = self._N

        return self._get_sample_times(n_samples, t_0)

    @property
    def N(self):
        return self._N

    @property
    def dt(self):
        return self._dt

    @property
    def sr(self):
        return self._sr

    @N.setter
    def N(self, val):
        self._N = val

    @dt.setter
    def dt(self, val):
        self._dt = val
        self._sr = 1/val

    @sr.setter
    def sr(self, val):
        self._sr = val
        self._dt = 1/val
