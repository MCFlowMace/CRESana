

"""

Author: F. Thomas
Date: August 11, 2022

"""

__all__ = []

import numpy as np
from abc import ABC, abstractmethod

from .utility import norm_squared, Interpolator2dx, differentiate
from .physicsconstants import speed_of_light
    

class RetardedSimCalculator(ABC):
    
    def __init__(self, positions):
        
        self.positions = positions

    @abstractmethod
    def __call__(self, t_sample, electron_simulator):
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
        
        t, coords = electron_simulator.get_sample_time_trajectory(t_sample)
        
        d_vec, d = self.calc_d_vec_and_abs(coords)
        
        t_ret_initial = self.get_retarded_time(t, d)
                                                
        retarded_electron_sim = electron_simulator(t_ret_initial)
        
        d_vec, d = self.calc_d_vec_and_abs(retarded_electron_sim.coords)
                                            
        return retarded_electron_sim, t, d_vec, d
        
    def get_retarded_time(self, t, d):

        def t_ret_0(t, d):
            
            print('Calculating 0th order taylor retarded time')
            t_travel = d/speed_of_light
            t_ret = t - t_travel

            return t_ret

        def t_ret_1(t, d):
            
            print('Calculating 1st order taylor retarded time')
            v = differentiate(d, t)
            t_travel = d/(speed_of_light+v)
            t_ret = t - t_travel

            return t_ret

        def t_ret_2(t, d):
            
            print('Calculating 2nd order taylor retarded time')
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
    
    def __call__(self, t_sample, electron_simulator):

        t_ret = self.get_retarded_time(t_sample, electron_simulator)
        ind_non_causal = t_ret<0
        t_ret[ind_non_causal] = 0.

        retarded_electron_sim = electron_simulator(t_ret)
        
        d_vec, d = self.calc_d_vec_and_abs(retarded_electron_sim.coords)
        
        d[ind_non_causal] = np.inf #makes the received power 0 later on -> no signal before the radiation arrives
        return retarded_electron_sim, t_sample, d_vec, d

    def get_retarded_time(self, t_sample, electron_simulator):
        
        print('Calculating forward retarded time')

        d_vec, d = self.calc_d_vec_and_abs(electron_simulator.electron_sim.coords)

        t_travel = d/speed_of_light
        t_antenna = electron_simulator.electron_sim.t + t_travel
        
        t_ret_f = Interpolator2dx(t_antenna, electron_simulator.electron_sim.t)
        
        t_ret = t_ret_f(t_sample)
        
        return t_ret
        
