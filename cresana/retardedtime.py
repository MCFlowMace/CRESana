

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
    
    def __init__(self, positions, order=0, interpolation='nearest'):
        RetardedSimCalculator.__init__(self, positions)
        
        if order<0 or order>2:
            raise ValueError('Only taylor orders 0<=order<=2 allowed')
            
        if interpolation != 'nearest' and interpolation != 'spline':
            raise ValueError('interpolation must be either "nearest" or "spline"')
            
        self.order = order
        self.interpolation = interpolation
        
    def __call__(self, t_sample, electron_sim):
        
        t, coords = self.get_sample_time_trajectory(t_sample, electron_sim)
        
        d_vec, d = self.calc_d_vec_and_abs(coords)
        
        t_ret_initial = self.get_retarded_time(t, d)
                                                
        retarded_electron_sim = self.get_retarded_sim(electron_sim, t_ret_initial)
        
        d_vec, d = self.calc_d_vec_and_abs(retarded_electron_sim.coords)
                                            
        return retarded_electron_sim, t, d_vec, d
        
    def enforce_causality(self, t_ret_initial, t_ret, B_ret):
        
        # setting to zero adds a small error for the initial phase in the phase integral
        # since this way the integral starts from t_ret=0 (which is correct) but 
        # it is assumed that omega(t_ret=0) = 0
        # the alternative solution of passing only the causal indices to the integral
        # has a wrong initial phase as well since it starts the integral from
        # some t_ret>0. The correct solution would be to find the correct value of omega(t_ret=0)
        # and integrating from there
        ind_non_causal = t_ret_initial<0
        B_ret[ind_non_causal] = 0
        t_ret[ind_non_causal] = 0
        
    def get_retarded_sim(self, electron_sim, t_ret_initial):
        
        if self.interpolation=='spline':

            B_f = interp1d(electron_sim.t, electron_sim.B_vals, kind='cubic', 
                            bounds_error=False, fill_value='extrapolate')
            pitch_f = interp1d(electron_sim.t, electron_sim.pitch, kind='cubic', 
                            bounds_error=False, fill_value='extrapolate')
            E_f = interp1d(electron_sim.t, electron_sim.E_kin, kind='cubic', 
                            bounds_error=False, fill_value='extrapolate')
            
            t_ret = t_ret_initial
            B_ret = B_f(t_ret)
            pitch_ret = pitch_f(t_ret)
            E_ret = E_f(t_ret)
            coords_ret = coords_f(t_ret)
            
        else:

            t_ret, sample_ind = find_nearest_samples2d(t_ret_initial, electron_sim.t)
            B_ret = electron_sim.B_vals[sample_ind]
            pitch_ret = electron_sim.pitch[sample_ind]
            E_kin_ret = electron_sim.E_kin[sample_ind]
            coords_ret = electron_sim.coords[sample_ind]
        
        self.enforce_causality(t_ret_initial, t_ret, B_ret)
        
        return ElectronSim(coords_ret, t_ret, B_ret, E_kin_ret, 
                                    pitch_ret, electron_sim.B_direction)
        
    def get_sample_time_trajectory(self, t_sample, electron_sim):
        
        if self.interpolation=='spline':
            t = t_sample
            self.coords_f = interp1d(electron_sim.t, electron_sim.coords, 
                                kind='cubic', axis=0, bounds_error=False, fill_value='extrapolate')
            coords = coords_f(t)
        else:
            #nearest neighbor interpolation
            t, sample_ind = find_nearest_samples(t_sample, electron_sim.t)
            coords = electron_sim.coords[sample_ind]
        
        return t, coords
    
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
    
    def __init__(self, positions, compression=0.5):
        
        RetardedSimCalculator.__init__(self, positions)
        
        if compression!='None' and compression > 1.0:
            print('Warning: using compression>1.0 is discouraged')
            
        self.compression = compression
        
    def get_decimation_factor(self, t_traj, t_sample):
        dt_high = t_traj[1]-t_traj[0]
        dt_low = t_sample[1] - t_sample[0]
        c = dt_high/dt_low if self.compression=='None' else self.compression
        
        decimation_factor = int(dt_low/dt_high*c)
        
        return decimation_factor
    
    def get_undersampled(self, electron_sim, decimation_factor):
        
        t = electron_sim.t[::decimation_factor]
        pitch = electron_sim.pitch[::decimation_factor]
        B = electron_sim.B_vals[::decimation_factor]
        coords = electron_sim.coords[::decimation_factor]
        E_kin = electron_sim.E_kin[::decimation_factor]
        
        return ElectronSim(coords, t, B, E_kin, 
                                    pitch, electron_sim.B_direction)
    
    def __call__(self, t_sample, electron_sim):
        
        decimation_factor = get_decimation_factor(electron_sim.t, t_sample)
        
        electron_sim_undersampled = self.get_undersampled(electron_sim, decimation_factor)
        
        B_ret_f = interp1d(electron_sim_undersampled.t, 
                            electron_sim_undersampled.B_vals, kind='cubic')
        pitch_ret_f = interp1d(electron_sim_undersampled.t, 
                            electron_sim_undersampled.pitch, kind='cubic')
        E_ret_f = interp1d(electron_sim_undersampled.t, 
                            electron_sim_undersampled.E_kin, kind='cubic')
        coords_ret_f = interp1d(electron_sim_undersampled.t, 
                            electron_sim_undersampled.coords, kind='cubic', axis=0)
        
        
        #d_vec = np.expand_dims(test_pos,1) - coords
        #d = np.sqrt(np.sum(d_vec**2, axis=-1))
        
        d_vec, d = self.calc_d_vec_and_abs(retarded_electron_sim.coords)
        
        t_travel = d/speed_of_light
        t_antenna = t_traj + t_travel
        
        
        causal = np.expand_dims(t_sample,0)>=np.expand_dims(t_antenna[:,0],1)
        
        t_ret_f = Interpolator2dx(t_antenna, t_traj)
        
        t_ret = t_ret_f(t_sample)
        
        t_ret[~causal] = 0.

        B_ret = np.zeros(t_ret.shape)
        pitch_ret = np.zeros(t_ret.shape)
        E_ret = np.zeros(t_ret.shape)
        
        coords_ret = np.zeros(t_ret.shape + (3,))
        
        d_ret = np.zeros_like(coords_ret)
        
        B_ret[causal] = B_ret_f(t_ret[causal])
        pitch_ret[causal] = pitch_ret_f(t_ret[causal])
        E_ret[causal] = E_ret_f(t_ret[causal])
        coords_ret[causal] = coords_ret_f(t_ret[causal])

        #d_ret = np.expand_dims(test_pos, 1) - coords_ret
        
        d_vec, d = self.calc_d_vec_and_abs(retarded_electron_sim.coords)
        
        ElectronSim(coords_ret, t_ret, B_ret, E_kin_ret, 
                                    pitch_ret, electron_sim.B_direction)
        
        return retarded_electron_sim, t_sample, d_vec, d
