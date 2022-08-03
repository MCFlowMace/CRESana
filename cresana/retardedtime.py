

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

import numpy as np
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod


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
        
    def calc_polar_angle(self, r_norm, B_direction):

        return np.arccos(np.dot(r_norm, B_direction))


class TaylorRetardedSimCalculator(RetardedSimCalculator):
    
    def __init__(self, positions):
        RetardedSimCalculator.__init__(positions)
        
    def __call__(self, t_sample, electron_sim):
        
        t, coords = self.get_sample_time_trajectory(N, electron_sim)
        
        d_vec, d = self.calc_d_vec_and_abs(coords)
        theta = self.calc_polar_angle(d_vec, electron_sim.B_direction)
        
        t_ret_initial = self.get_retarded_time(t, d)


        t_ret, B_ret, pitch_ret, E_kin_ret = self.get_sampled_model_parameters(electron_sim, t_ret_initial)

                                                
        retarded_electron_sim = ElectronSim(coords, t_ret, B_ret, E_kin_ret, 
                                            pitch_ret, electron_sim.B_direction)
        
    def get_sample_time_trajectory(self, t_sample, electron_sim):
        t, sample_ind = find_nearest_samples(t_sample, electron_sim.t)
        coords = electron_sim.coords[sample_ind]
        
        return t, coords
    
    def get_retarded_time(self, t, d):

        t_travel = d/speed_of_light
        t_ret = t - t_travel
        
        return t_ret
        
    def get_retarded_time_0(t, d):
        t_travel = d/speed_of_light
        t_ret = t - t_travel

        return t_ret

    def get_retarded_time_1(t, d):
        
        v = differentiate(d, t)
        t_travel = d/(speed_of_light+v)
        t_ret = t - t_travel

        return t_ret

    def get_retarded_time_2(t, d):
        
        v = differentiate(d, t)
        a = differentiate(v, t)
        
        t_ret = t - (speed_of_light + v)/a + np.sqrt((speed_of_light + v)**2 - 2*a*d)/a
        
        t_ret[a==0.] = -1.

        return t_ret
    
    #~ def calc_antenna_dist(self, coords):
        #~ self.dist_cache = self.antenna_array.get_distance(coords)
        #~ self.d_abs_cache = np.sqrt(np.sum(self.dist_cache**2, axis=-1))
        
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
        E_kin = electron_sim.E_kin[sample_ind_correct]
        
        self.enforce_causality(t_retarded, t_ret_correct, B_sample, pitch)
        
        return t_ret_correct, B_sample, pitch, E_kin
