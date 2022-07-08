

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

import numpy as np


def calculate_received_power(P_transmitted, D_transmitter, w_transmitter, D_receiver, d_squared):
    """
    Friis transmission equation
    """
    P_received = np.empty(shape=P_transmitted.shape)
    P_received = P_transmitted*D_transmitter*D_receiver*np.pi*speed_of_light**2/d_squared
    
    ind = w_transmitter != 0
    P_received[ind] /= w_transmitter[ind]**2
    P_received[np.invert(ind)] = 0
    return P_received
    
    
class AntennaArray:

    def __init__(self, positions, directive_gain_function, resistance=390):
        #attention !!! orientation of antenna is NOT included
        self.positions = positions
        self.directive_gain_function = directive_gain_function
        self.resistance = resistance
        
    def get_distance(self, pos):
        return pos - np.expand_dims(self.positions, 1)
        
    def power_to_voltage(self, P):
        return np.sqrt(P*ev*self.resistance)
    
    def get_detected_power(self, dist, P_transmitted, D_transmitter, w_transmitter):
        
        D_receiver = self.directive_gain_function(dist)
        d_squared = np.sum(dist**2, axis=-1)
        
        return calculate_received_power(P_transmitted, D_transmitter, 
                                        w_transmitter, D_receiver, d_squared)
        
    def get_amplitude(self, dist, P_transmitted, D_transmitter, w_transmitter):
        P_received = self.get_detected_power(dist, P_transmitted, D_transmitter, 
                                                w_transmitter)
        return self.power_to_voltage(P_received)
        
    @classmethod
    def make_multi_ring_array(cls, R, n_antenna, n_rings, z_min, z_max, gain_f, resistance=390):

        z = np.linspace(z_min, z_max, n_rings)

        angles = np.linspace(0, 1, n_antenna, endpoint=False)*2*np.pi
        x = np.cos(angles)*R
        y = np.sin(angles)*R

        xx, zz = np.meshgrid(x, z)
        yy, _ = np.meshgrid(y, z)

        positions = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))

        instance = cls(positions, gain_f, resistance)

        return instance

