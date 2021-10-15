

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

import numpy as np
from .physicsconstants import speed_of_light, ev
from .cyclotronmotion import get_omega_cyclotron, get_slope, get_radiated_power, get_omega_cyclotron_time_dependent
from .electronsim import simulate_electron
from scipy.integrate import cumtrapz

import matplotlib.pyplot as plt

class Sampler:

    def __init__(self, sr, N=8192):

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

def get_signal(A, phase):

    return A*np.exp(1.0j*phase)

def get_cyclotron_phase(t, w_avg, w_m):

    return t*(w_avg - w_m)

def get_energy_loss_phase(t, slope):

    return 0.5*slope*t**2

def get_phase_shift(d, w):

    return w*d/speed_of_light

# ~ def get_distance(pos_e, pos_antenna):

    # ~ return np.sqrt(np.sum((pos_e-pos_antenna)**2, axis=1))

def get_cyclotron_phase_int(w, t):

    return cumtrapz(w, x=t, initial=0.0)

def find_nearest_samples(t1, t2):

    ind = np.searchsorted((t2[1:]+t2[:-1])/2, t1)
    last = np.searchsorted(ind, t2.shape[0]-1)

    return t2[ind[:last]], ind[:last]

def power_to_voltage(P):

    resistance = 390 # 50 #ohm
    return np.sqrt(P*resistance)

class SignalModel:

    def __init__(self, antenna_array, sr, w_mix, AM=True, slope=True, frequency_weight=lambda w: 1.0):

        self.sampler = Sampler(sr)
        self.antenna_array = antenna_array
        self.w_mix = w_mix
        self.AM = AM
        self.slope = slope
        self.frequency_weight = frequency_weight

    def get_samples(self, N, electron_sim):

        t_sample = self.sampler(N)

        t_traj = electron_sim.t
        d_antenna = self.antenna_array.get_distance(electron_sim.coords)
        d_antenna_abs = np.sqrt(np.sum(d_antenna**2, axis=-1))
        t_antenna = t_traj + d_antenna_abs/speed_of_light

        causal_ind = (t_sample-t_antenna[0,0])>0
        t_causal = t_sample[causal_ind]
        t_sample_mod, ind = find_nearest_samples(t_causal, t_antenna[0])

        print(d_antenna.shape)
        print(ind.shape)

        t_ret = t_traj[ind]

        E_kin = electron_sim.E_kin
        B_sample = electron_sim.B_vals[ind]
        theta = electron_sim.theta[ind]
        d_antenna = d_antenna[:,ind]
        #d_antenna_abs =

        A = np.zeros((d_antenna.shape[0], t_sample.shape[0]))
        phase = np.zeros((d_antenna.shape[0], t_sample.shape[0]))

        power = get_radiated_power(E_kin, theta, B_sample)
        w = get_omega_cyclotron_time_dependent(B_sample, E_kin, power, t_ret)

        gain = self.antenna_array.get_amplitude(d_antenna)

        print(t_sample.shape)
        print(gain.shape)
        print(w.shape)
        print(t_sample_mod.shape)
        print(A.shape)

        detected_power = gain*power

        first_causal = t_sample.shape[0] - t_causal.shape[0]
        print(first_causal)

        A[:,first_causal:] = power_to_voltage(detected_power*ev)


        phase[:,first_causal:] = get_cyclotron_phase_int(w, t_ret)

        phase -= self.w_mix*t_sample


        return get_signal(A, phase)

    def get_samples_from_electron(self, N, electron, trap):

        electron_sim = simulate_electron(electron, self.sampler, trap, N)

        return self.get_samples(N, electron_sim)
