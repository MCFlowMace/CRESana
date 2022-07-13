

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

import numpy as np

from .utility import normalize, project_on_plane, angle_with_orientation


def calculate_received_power(P_transmitted, w_transmitter, G_receiver, d_squared):
    """
    Friis transmission equation
    P_transmitted is power of transmitting antenna*gain of transmitter
    """
    P_received = np.empty(shape=P_transmitted.shape)
    P_received = P_transmitted*G_receiver*speed_of_light**2/(4*d_squared)
    
    ind = w_transmitter != 0
    P_received[ind] /= w_transmitter[ind]**2
    P_received[np.invert(ind)] = 0
    return P_received
    

def slot_directivity_factor(theta, phi):
    
    theta_factor = np.zeros_like(theta)
    
    nonzero = (theta != 0.0)&(np.abs(theta) != np.pi)
    
    print(nonzero)
    
    theta_factor[nonzero] = np.cos(np.pi/2*np.cos(theta[nonzero]))/np.sin(theta[nonzero])
    phi_factor = np.cos(phi)
    
    return theta_factor*phi_factor
    
    
def isotropic_directivity_factor(theta, phi):
    
    return 1.0
    
    
class AntennaArray:

    def __init__(self, positions, normals, polarizations, transfer_function, directivity_function, resistance=50):

        self.positions = positions
        self.normals = normals
        self.polarizations = polarizations
        self.resistance = resistance
        self.transfer_function = transfer_function
        self.directivity_function = directivity_function
        self.cross_polarizations = normalize(np.cross(normals, polarizations))
        
    def power_to_voltage(self, P):
        #P = u_rms^2/R and u0*sqrt(2)=u_rms for a sine wave -> u0 = sqrt(2 * P * R)
        return np.sqrt(2*P*self.resistance)
                                 
    def get_phase_shift(self):
        return 
        
    def get_polarization_mismatch_gain(self, pol_x, pol_y, delta_phase):
    
        a = np.einsum('ik,ijk->ij', self.polarizations, pol_x) #(dot(pol_x, polarizations))
        b = np.einsum('ik,ijk->ij', self.polarizations, pol_y)
        ab = 2*np.cos(delta_phase)*a*b
        
        return a**2 + b**2 + ab
        
    def get_directional_gain(self, d_vec):
        
        theta, phi = self.get_gain_angles(d_vec)
        
        return self.directivity_function(theta, phi)**2
        
    def get_gain_angles(self, d_vec):
        
        r_project_pol = project_on_plane(-d_vec, self.cross_polarizations)
        r_project_cross = project_on_plane(-d_vec, self.polarizations)
        
        theta = angle_with_orientation(np.expand_dims(self.normals, 1), r_project_pol, 
                                       np.expand_dims(self.cross_polarizations, 1))

        phi = angle_with_orientation(np.expand_dims(self.normals, 1), r_project_cross, 
                                       np.expand_dims(self.polarizations, 1))
        
        return theta, phi
        
    def get_tf_gain(self, w_receiver):
        
        eta = 377 #impedance of free space
        
        tf = np.abs(self.transfer_function(w_receiver))
        
        return tf**2*eta*w_receiver**2/(np.pi*speed_of_light**2*self.resistance)
        
    def get_receiver_gain(self, pol_x, pol_y, d_vec, w_receiver):
        polarization_mismatch_gain = self.get_polarization_mismatch_gain(pol_x, pol_y, np.pi/2)
        directional_gain = self.get_directional_gain(d_vec)
        tf_gain = self.get_tf_gain(w_receiver)
        #IQ_receiver_gain = 1 # = A_LO**2 with A_LO=1
        LPF_gain = 0.5 #LPF after LO cuts half the spectrum
        
        return polarization_mismatch_gain*directional_gain*tf_gain*LPF_gain
        
    def get_amplitude(self, dist, P_transmitted, w_transmitter, w_receiver, pol_x, pol_y, d_vec, d):
        G_receiver = self.get_receiver_gain()
        P_received = calculate_received_power(P_transmitted, w_transmitter, G_receiver, d**2)
        return self.power_to_voltage(P_received)
        
    @classmethod
    def make_multi_ring_array(cls, R, n_antenna, n_rings, z_min, z_max, 
                                transfer_function, directivity_function, resistance=50):

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
        polarizations[:, 0] = positions[:,1]/R
        polarizations[:, 1] = -positions[:,0]/R

        instance = cls(positions, normals, polarizations, transfer_function, directivity_function, resistance)

        return instance

