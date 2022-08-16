

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

import numpy as np
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod

from .utility import norm_squared, Interpolator2dx, differentiate
from .physicsconstants import speed_of_light
from .electronsim import ElectronSim


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
    

class RetardedSimCalculator(ABC):
    
    def __init__(self, positions):
        
        self.positions = positions

    @abstractmethod
    def __call__(self, t_sample, electron_sim):
        pass
        
    def calc_d_vec_and_abs(self, coords):
        
        r = np.expand_dims(self.positions, 1) - coords
        
        d = np.sqrt(norm_squared(r))

        d_vec = r/d
        
        return d_vec, d[...,0]


class TaylorRetardedSimCalculator(RetardedSimCalculator):
    
    def __init__(self, positions, order=0):
        RetardedSimCalculator.__init__(self, positions)
        
        if order<0 or order>2:
            raise ValueError('Only taylor orders 0<=order<=2 allowed')
            
        self.order = order
        
    def __call__(self, t_sample, electron_simulator):
        
        #t, coords = self.get_sample_time_trajectory(t_sample, electron_sim)
        t, coords = electron_simulator.get_sample_time_trajectory(t_sample)
        
        d_vec, d = self.calc_d_vec_and_abs(coords)
        
        t_ret_initial = self.get_retarded_time(t, d)
                                                
        retarded_electron_sim = electron_simulator(t_ret_initial) #self.get_retarded_sim(electron_sim, t_ret_initial)
        
        d_vec, d = self.calc_d_vec_and_abs(retarded_electron_sim.coords)
                                            
        return retarded_electron_sim, t, d_vec, d
        
    def get_retarded_time(self, t, d):

        def t_ret_0(t, d):
            t_travel = d/speed_of_light
            t_ret = t - t_travel

            return t_ret

        def t_ret_1(t, d):
            
            v = differentiate(d, t)
            t_travel = d/(speed_of_light+v)
            t_ret = t - t_travel

            return t_ret

        def t_ret_2(t, d):
            
            v = differentiate(d, t)
            a = differentiate(v, t)
            
            t_ret = t - (speed_of_light + v)/a + np.sqrt((speed_of_light + v)**2 - 2*a*d)/a
            
            t_ret[a==0.] = -1.

            return t_ret
            
        if self.order == 0:
            return t_ret_0(t, d)
        elif self.order == 1:
            return t_ret_1(t,d)
        else:
            return t_ret_2(t,d)
        

class ForwardRetardedSimCalculator(RetardedSimCalculator):
    
    def __init__(self, positions):
        
        RetardedSimCalculator.__init__(self, positions)
        
        #~ if compression!='None' and compression > 1.0:
            #~ print('Warning: using compression>1.0 is discouraged')
            
        #~ self.compression = compression
        
    #~ def get_decimation_factor(self, t_traj, t_sample):
        #~ dt_high = t_traj[1]-t_traj[0]
        #~ dt_low = t_sample[1] - t_sample[0]
        #~ c = dt_high/dt_low if self.compression=='None' else self.compression
        
        #~ decimation_factor = int(dt_low/dt_high*c)
        
        #~ return decimation_factor
    
    #~ def get_undersampled(self, electron_sim, decimation_factor):
        
        #~ t = electron_sim.t[::decimation_factor]
        #~ pitch = electron_sim.pitch[::decimation_factor]
        #~ B = electron_sim.B_vals[::decimation_factor]
        #~ coords = electron_sim.coords[::decimation_factor]
        #~ E_kin = electron_sim.E_kin[::decimation_factor]

        
        #~ return ElectronSim(coords, t, B, E_kin, 
                                    #~ pitch, electron_sim.B_direction)
    
    def __call__(self, t_sample, electron_simulator):
        
       # decimation_factor = self.get_decimation_factor(electron_sim.t, t_sample)
        
       # electron_sim_undersampled = self.get_undersampled(electron_sim, decimation_factor)
        
        t_ret = self.get_retarded_time(t_sample, electron_simulator)
        
        #~ causal = np.expand_dims(t_sample,0)>=np.expand_dims(t_antenna[:,0],1)
        #~ t_ret[~causal] = 0.

        #~ B_ret = np.zeros(t_ret.shape)
        #~ pitch_ret = np.zeros(t_ret.shape)
        #~ E_ret = np.zeros(t_ret.shape)
        
        #~ coords_ret = np.zeros(t_ret.shape + (3,))
        
        #~ d_ret = np.zeros_like(coords_ret)
        
        #~ B_ret[causal] = B_ret_f(t_ret[causal])
        #~ pitch_ret[causal] = pitch_ret_f(t_ret[causal])
        #~ pitch_ret[~causal] = np.pi/2
        #~ E_ret[causal] = E_ret_f(t_ret[causal])
        #~ E_ret[~causal] = 1.0 #E = 0 causes divide by zero errors
        #~ coords_ret[causal] = coords_ret_f(t_ret[causal])

        retarded_electron_sim = electron_simulator(t_ret_initial)
        
        d_vec, d = self.calc_d_vec_and_abs(retarded_electron_sim.coords)
        
        return retarded_electron_sim, t_sample, d_vec, d

    def get_retarded_time(self, t_sample, electron_simulator):

        d_vec, d = self.calc_d_vec_and_abs(electron_simulator.electron_sim.coords)

        t_travel = d/speed_of_light
        t_antenna = electron_simulator.electron_sim.t + t_travel
        
        t_ret_f = Interpolator2dx(t_antenna, electron_simulator.electron_sim.t)
        
        t_ret = t_ret_f(t_sample)
        
        return t_ret
        
