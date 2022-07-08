

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

    def __init__(self, positions, normals, polarizations, resistance=50):

        self.positions = positions
        self.normals = normals
        self.polarizations = polarizations
        self.resistance = resistance
        
    def power_to_voltage(self, P):
        return np.sqrt(P*self.resistance)
    
    #~ def get_detected_power(self, dist, P_transmitted, D_transmitter, w_transmitter):
        
        #~ D_receiver = self.directive_gain_function(dist)
        #~ d_squared = np.sum(dist**2, axis=-1)
        
        #~ return calculate_received_power(P_transmitted, D_transmitter, 
                                        #~ w_transmitter, D_receiver, d_squared)
                                        
    def get_phase_shift(self):
        return 
        
    def get_amplitude(self, dist, P_transmitted, D_transmitter, w_transmitter):
        P_received = self.get_detected_power(dist, P_transmitted, D_transmitter, 
                                                w_transmitter)
        u_rms = self.power_to_voltage(P_received)
        
        #u_rms = u0/sqrt(2) for a sine wave
        return np.sqrt(2)*u_rms
        
    @classmethod
    def make_multi_ring_array(cls, R, n_antenna, n_rings, z_min, z_max, resistance=50):

        z = np.linspace(z_min, z_max, n_rings)

        angles = np.linspace(0, 1, n_antenna, endpoint=False)*2*np.pi
        x = np.cos(angles)*R
        y = np.sin(angles)*R

        xx, zz = np.meshgrid(x, z)
        yy, _ = np.meshgrid(y, z)

        positions = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
        normals = np.zeros_like(positions)
        
        normals[:,0] = -positions[:,0]/R
        normals[:,1] = -positions[:,1]/R
        
        #set polarization such that at position = (R, 0, 0) -> pol = (0, -1, 0)
        polarizations = np.zeros_like(positions)
        polarizations[:, 0] = -positions[:,1]/R
        polarizations[:, 1] = -positions[:,0]/R

        instance = cls(positions, normals, polarizations, resistance)

        return instance

