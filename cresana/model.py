

"""

Author: F. Thomas
Date: May 17, 2023

"""

from abc import ABC, abstractmethod

from .electronsim import Electron, AnalyticSimulation
from .sampling import Simulation


class CRESanaModel(ABC):

    def __init__(self, sr, f_LO, n_samples, flattened=True):
        self.sr = sr
        self.dt = 1/sr
        self.f_LO = f_LO
        self.flattened = flattened
        self.n_samples = n_samples

    @property
    @abstractmethod
    def array(self):
        pass

    @property
    @abstractmethod
    def trap(self):
        pass

    def __call__(self, E_kin, pitch, r, t0, tau):
        z0 = 0.0
        electron = Electron(E_kin, pitch, t_start=t0, t_len=tau, r=r, z0=z0)
        data = self._simulate(electron)

        if self.flattened:
            data = data.flatten()

        return data
    
    def check_sample_time(self, electron):
        samples_required = (electron.t_start + electron.t_len)/self.dt

        if self.n_samples < samples_required:
            raise ValueError(f'Too few samples, electron signal cannot be sampled to the end! You need at least {samples_required} \
                             samples plus some margin to account for the additional delay time and roundoff error.')

    def _simulate(self, electron):
        self.check_sample_time(electron)
        return self.simulate(electron)

    def simulate(self, electron):

        t_max = self.dt*self.n_samples

        sim = AnalyticSimulation(self.trap, electron, 2*self.n_samples, t_max)

        simulation = Simulation(self.array, self.sr, self.f_LO)
        return simulation.get_samples(self.n_samples, sim)
