

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

from .physicsconstants import speed_of_light, ev
from .cresphysics import get_radiated_power, get_omega_cyclotron_time_dependent, get_directive_gain
from .electronsim import simulate_electron


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
    
    
def find_nearest_samples2d(t1, t2):
    t = np.empty(shape=t1.shape)
    ind = np.empty(shape=t1.shape, dtype=np.int64)
    
    for i in range(t1.shape[0]):
        t[i], ind[i] = find_nearest_samples(t1[i], t2)
        
    return t, ind


class SignalModel:

    def __init__(self, antenna_array, sr, w_mix, AM=True, slope=True, frequency_weight=lambda w: 1.0):
        self.sampler = Sampler(sr)
        self.antenna_array = antenna_array
        self.w_mix = w_mix
        self.AM = AM
        self.slope = slope
        self.frequency_weight = frequency_weight
        
    def get_sample_time_trajectory(self, N, electron_sim):
        t, sample_ind = find_nearest_samples(self.sampler(N), electron_sim.t)
        coords = electron_sim.coords[sample_ind]
        
        return t, coords
    
    def get_retarded_time(self, t, coords):
        #dist = self.antenna_array.get_distance(coords)
        t_travel = self.d_abs_cache/speed_of_light
        t_ret = t - t_travel
        
        return t_ret
    
    def calc_antenna_dist(self, coords):
        self.dist_cache = self.antenna_array.get_distance(coords)
        self.d_abs_cache = np.sqrt(np.sum(self.dist_cache**2, axis=-1))
        
    def enforce_causality(self, t_retarded, t_ret_correct, B_sample, pitch):
        
        # setting to zero adds a small error for the initial phase in the phase integral
        # since this way the integral starts from t_ret=0 (which is correct) but 
        # it is assumed that omega(t_ret=0) = 0
        # the alternative solution of passing only the causal indices to the integral
        # has a wrong initial phase as well since it starts the integral from
        # some t_ret>0. The correct solution would be to find the correct value of omega(t_ret=0)
        # and integrating from there
        ind_non_causal = t_retarded<0
        B_sample[ind_non_causal] = 0
        t_ret_correct[ind_non_causal] = 0
        
    def get_sampled_model_parameters(self, electron_sim, t_retarded):

        t_ret_correct, sample_ind_correct = find_nearest_samples2d(t_retarded, electron_sim.t)
        B_sample = electron_sim.B_vals[sample_ind_correct]
        pitch = electron_sim.pitch[sample_ind_correct]
        
        self.enforce_causality(t_retarded, t_ret_correct, B_sample, pitch)
        
        return t_ret_correct, B_sample, pitch

    def get_samples(self, N, electron_sim):
        
        t, coords = self.get_sample_time_trajectory(N, electron_sim)
        
        self.calc_antenna_dist(coords)

        t_ret = self.get_retarded_time(t, coords)

        t_ret_correct, B_sample, pitch = self.get_sampled_model_parameters(electron_sim, t_ret)

        radiated_power = get_radiated_power(electron_sim.E_kin, pitch, B_sample)
        w = get_omega_cyclotron_time_dependent(B_sample, electron_sim.E_kin, 
                                                radiated_power, t_ret_correct)
        
        phase = get_cyclotron_phase_int(w, t_ret_correct)
        
        angle = np.arccos(np.dot(-self.dist_cache, electron_sim.B_direction)/self.d_abs_cache)
        
        directive_gain = get_directive_gain(electron_sim.E_kin, pitch, angle)
        A = self.antenna_array.get_amplitude(self.dist_cache, radiated_power, 
                                                directive_gain, w)

        phase -= self.w_mix*t

        return get_signal(A, phase)
        
    def get_samples_from_electron(self, N, electron, trap):
        electron_sim = simulate_electron(electron, self.sampler, trap, N)

        return self.get_samples(N, electron_sim)

