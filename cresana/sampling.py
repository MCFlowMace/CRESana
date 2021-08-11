

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

class Sampler:

    def __init__(self, sr):

        self._sr = sr
        self._dt = 1/sr

    def _get_sample_times(self, N, t_0):

        return np.arange(N)*self._dt + t_0

    def __call__(self, N, t_0=0):

        return self._get_sample_times(N, t_0)
