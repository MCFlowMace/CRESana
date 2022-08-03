

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

import numpy as np
from scipy.integrate import cumtrapz

from .electronsim import simulate_electron
from .cyclotronphysics import AnalyticCyclotronField
from .retardedtime import TaylorRetardedSimCalculator, ForwardRetardedSimCalculator


class Clock:
    
    def __init__(self, frequency):

        self._f = frequency
        self._dt = 1/frequency
        
    def _get_sample_times(self, N, t_0):
        return np.arange(N)*self._dt + t_0
        
    def __call__(self, N, t_0=0.):

        return self._get_sample_times(N, t_0)
        
    @property
    def dt(self):
        return self._dt
        
    @property
    def f(self):
        return self._f
        
    @dt.setter
    def dt(self, val):
        self._dt = val
        self._sr = 1/val
        
    @f.setter
    def f(self, val):
        self._f = val
        self._dt = 1/val
        
    
class IQReceiver:
    
    def __init__(self, f_LO, phase=np.pi/2, A_LO=1.0):
        
        self.w_LO = 2*np.pi*f_LO
        self.phase = phase
        self.A_LO = A_LO
        
    def __call__(self, t, antenna_array, received_copolar_field_power, 
                    field_phase, d_vec):
        
        field_phase -= self.w_LO*t
        
        samples = antenna_array.get_voltage(received_copolar_field_power, 
                                                field_phase, 
                                                self.w_LO, d_vec, 
                                                real_signal=False)
                                                
        return self.A_LO*samples*np.exp(1.0j*self.phase)


class Simulation:

    def __init__(self, antenna_array, sampling_rate, f_LO, **kwargs):
        self.clock = Clock(sampling_rate)
        self.antenna_array = antenna_array
        self.receiver = IQReceiver(f_LO)
        self.configure(kwargs)
        
    def configure(self, config_dict):
        
        #default configuration
        self.use_AM = True
        self.use_FM = True
        use_taylor = True
        taylor_order = 0
        use_interpolation = False
        interpolation_compression = 0.5
        
        if 'use_AM' in config_dict:
            self.use_AM = config_dict['use_AM']
            
        if 'use_FM' in config_dict:
            self.use_FM = config_dict['use_FM']
            
        if 'use_taylor' in config_dict:
            use_taylor = config_dict['use_taylor']
            
        if 'use_interpolation' in config_dict:
            use_interpolation = config_dict['use_interpolation']
            
        if 'interpolation_compression' in config_dict:
            interpolation_compression = config_dict['interpolation_compression']
            
        if 'taylor_order' in config_dict:
            taylor_order = config_dict['taylor_order']
        
        if use_taylor: 
            interpolation = 'spline' if use_interpolation else 'nearest'
            self.retarded_calculator = TaylorRetardedSimCalculator(
                                            self.antenna_array.positions, 
                                            order=taylor_order, 
                                            interpolation=interpolation)
        else:
            self.retarded_calculator = ForwardRetardedSimCalculator(
                                            self.antenna_array.positions,
                                            compression=interpolation_compression)
            
    def get_cyclotron_phase_int(self, w, t):
        return cumtrapz(w, x=t, initial=0.0)

    def get_samples(self, N, electron_sim):
        
        t_sample = self.clock(N)
        
        retarded_electron_sim, t_sample, d_vec, d = self.retarded_calculator(t_sample, electron_sim)
        
        t_ret = retarded_electron_sim.t

        cyclotron_field = AnalyticCyclotronField(retarded_electron_sim)
        
        w, P_transmitted, pol_x, pol_y, phase = cyclotron_field.get_field_parameters(d_vec)
                                                
        received_copolar_field_power = self.antenna_array.get_received_copolar_field_power(P_transmitted, w, pol_x, pol_y, d)
        field_phase = self.get_cyclotron_phase_int(w, t_ret) + phase
        
        signal = self.receiver(t_sample, self.antenna_array, received_copolar_field_power, 
                                field_phase, d_vec)

        return signal


"""

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

"""
