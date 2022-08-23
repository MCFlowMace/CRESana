

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

import numpy as np
from scipy.integrate import cumtrapz

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
        taylor_order = None
        
        if 'use_AM' in config_dict:
            self.use_AM = config_dict['use_AM']
            
        if 'use_FM' in config_dict:
            self.use_FM = config_dict['use_FM']
            
        if 'use_taylor' in config_dict:
            use_taylor = config_dict['use_taylor']
            
        if 'taylor_order' in config_dict:
            taylor_order = config_dict['taylor_order']
        
        if taylor_order is not None: 
            self.retarded_calculator = TaylorRetardedSimCalculator(
                                            self.antenna_array.positions, 
                                            order=taylor_order)
        else:
            self.retarded_calculator = ForwardRetardedSimCalculator(
                                            self.antenna_array.positions)
            
    def get_cyclotron_phase_int(self, w, t):
        return cumtrapz(w, x=t, initial=0.0)

    def get_samples(self, N, electron_simulator):
        
        t_sample = self.clock(N)
        
        retarded_electron_sim, t_sample, d_vec, d = self.retarded_calculator(t_sample, electron_simulator)
        
        t_ret = retarded_electron_sim.t

        cyclotron_field = AnalyticCyclotronField(retarded_electron_sim, n_harmonic=1)
        
        w, P_transmitted, pol_x, pol_y, phase = cyclotron_field.get_field_parameters(d_vec)
                                                
        received_copolar_field_power = self.antenna_array.get_received_copolar_field_power(P_transmitted, w, pol_x, pol_y, d)
        field_phase = self.get_cyclotron_phase_int(w, t_ret) + phase
        
        signal = self.receiver(t_sample, self.antenna_array, received_copolar_field_power, 
                                field_phase, d_vec)

        return signal

