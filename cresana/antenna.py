

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import interp1d

from .utility import normalize, project_on_plane, angle_with_orientation
from .physicsconstants import speed_of_light


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


#voltage signal model is A*cos(phi) = A/2*(e^iphi + e^-iphi)
def get_signal(A, phase, real=False):
    signal = np.exp(1.0j*phase)
    
    if real:
        signal = signal + np.exp(-1.0j*phase)
        
    return A*0.5*signal
    
    
standard_impedance = 50.
free_space_impedance = 377.


class Antenna(ABC):
    
    def __init__(self):
        pass
        
    def get_tf_gain(self, w_receiver):
        
        tf = np.abs(self.transfer_function(w_receiver))
        
        return tf**2*free_space_impedance*w_receiver**2/(np.pi*speed_of_light**2*standard_impedance)
        
    def get_directivity_gain(self, theta, phi):
        return self.directivity_factor(theta, phi)**2
        
    def get_gain(self, theta, phi, w_receiver):
        g = self.get_directivity_gain(theta, phi)*self.get_tf_gain(w_receiver)
        return g
        
    def power_to_amplitude(self, P):
        #P = u_rms^2/R and u0*sqrt(2)=u_rms for a sine wave -> u0 = sqrt(2 * P * R)
        # u_rms = u0 for complex wave
        u0 =  np.sqrt(2*P*standard_impedance)
        return u0
        
    def get_phase(self, phase, w_receiver):
        
        angle = np.angle(self.transfer_function(w_receiver))
        
        return phase + angle
        
    def get_voltage(self, copolar_power, field_phase,
                    theta, phi, w_receiver, real_signal=False):
                        
        gain = self.get_gain(theta, phi, w_receiver)
        U0 = self.power_to_amplitude(copolar_power*gain)
        phase0 = self.get_phase(field_phase, w_receiver)
        element_signals = get_signal(U0, phase0, real=real_signal)
        
        return self.sum_elements(element_signals)
        
    @abstractmethod
    def transfer_function(self, w_receiver):
        pass
        
    @abstractmethod
    def directivity_factor(self, theta, phi):
        pass
        
    @abstractmethod
    def position_elements(self, positions):
        pass
        
    @abstractmethod
    def sum_elements(self, signals):
        pass
        

class IsotropicAntenna(Antenna):
    
    def __init__(self):
        
        Antenna.__init__(self)
        
    def transfer_function(self, w_receiver):
        
        tf_abs = np.sqrt(standard_impedance*speed_of_light**2*np.pi/(free_space_impedance*w_receiver**2))
        
        return tf_abs + 0.0j
        
    def directivity_factor(self, theta, phi):
        return np.ones_like(theta)
        
    def position_elements(self, positions):
        return positions
        
    def sum_elements(self, signals):
        return signals
        
    
class SlottedWaveguideAntenna(Antenna):
    
    def __init__(self, n_slots, tf_file_name, slot_offset=7.75e-3, 
                    tf_frequency_unit=1.0e9):
        
        Antenna.__init__(self)
        
        self.n_slots = n_slots
        self.slot_offset = slot_offset
        self._create_tf(tf_file_name, tf_frequency_unit)
        
    
    def _create_tf(self, tf_file_name, tf_frequency_unit=1.0e9):
        
        tf_data = np.loadtxt(tf_file_name)
        
        tf_f = tf_data[:,0]*2*np.pi*tf_frequency_unit
        tf_val_re = tf_data[:,1]
        tf_val_im = tf_data[:,2]
        
        inter_re = interp1d(tf_f, tf_val_re, kind='cubic')
        inter_im = interp1d(tf_f, tf_val_im, kind='cubic')
        
        self.interp_tf = lambda w: inter_re(w) + 1.0j*inter_im(w)
        
    def transfer_function(self, w_receiver):
        return self.interp_tf(w_receiver)
        
    def directivity_factor(self, theta, phi):
    
        theta_mod = np.pi/2 - theta
        
        theta_factor = np.zeros_like(theta_mod)
        
        nonzero = (theta_mod != 0.0)&(np.abs(theta_mod) != np.pi)
        
        theta_factor[nonzero] = np.cos(np.pi/2*np.cos(theta_mod[nonzero]))/np.sin(theta_mod[nonzero])
        phi_factor = np.cos(phi)
        
        return theta_factor*phi_factor
        
    def position_elements(self, positions):
        """
        Positions waveguide slots for given positions of Waveguide antennas
        """
        offset = np.zeros_like(positions)
        offset[:,2] = self.slot_offset
        
        n_offset = np.arange(self.n_slots)-self.n_slots/2 + 0.5
        
        offset_positions = n_offset*np.expand_dims(offset,-1)
        offset_positions_reshaped = np.einsum('ijk->kij', offset_positions).reshape((-1, 3))
        
        element_positions = np.tile(positions, (self.n_slots,1))
        
        return element_positions + offset_positions_reshaped
        
    def sum_elements(self, signals):
        
        signal_reshaped = signals.reshape((self.n_slots, -1, signals.shape[-1]))
        
        return np.sum(signal_reshaped, axis=0)
    
    
class AntennaArray:

    def __init__(self, positions, normals, polarizations, antenna):

        self.positions = positions
        self.normals = normals
        self.polarizations = polarizations
        self.cross_polarizations = normalize(np.cross(normals, polarizations))
        self.antenna = antenna
        
    def get_directivity_angles(self, d_vec):
        
        r_project_pol = project_on_plane(-d_vec, self.cross_polarizations)
        r_project_cross = project_on_plane(-d_vec, self.polarizations)
        
        theta = angle_with_orientation(np.expand_dims(self.normals, 1), r_project_pol, 
                                       np.expand_dims(self.cross_polarizations, 1))

        phi = angle_with_orientation(np.expand_dims(self.normals, 1), r_project_cross, 
                                       np.expand_dims(self.polarizations, 1))
        
        return theta, phi
        
    def get_polarization_mismatch(self, pol_x, pol_y, delta_phase):
    
        a = np.einsum('ik,ijk->ij', self.polarizations, pol_x) #(dot(pol_x, polarizations))
        b = np.einsum('ik,ijk->ij', self.polarizations, pol_y)
        ab = 2*np.cos(delta_phase)*a*b
        
        return a**2 + b**2 + ab
        
    def get_received_copolar_field_power(self, P_transmitted, w_transmitter, pol_x, pol_y, d):
        
        polarization_mismatch_gain = self.get_polarization_mismatch(pol_x, pol_y, np.pi/2)
        
        P_received = calculate_received_power(P_transmitted, w_transmitter, 1.0, d**2)
        
        return P_received
        
    def get_voltage(self, received_copolar_field_power, field_phase, 
                    w_receiver, d_vec, real_signal=False):
                        
        theta, phi = self.get_directivity_angles(d_vec)
        
        return self.antenna.get_voltage(received_copolar_field_power, 
                                        field_phase, theta, phi,
                                        w_receiver, real_signal=real_signal)
        
    @classmethod
    def make_multi_ring_array(cls, R, n_antenna, n_rings, z_min, z_max, 
                                antenna):

        z = np.linspace(z_min, z_max, n_rings)

        angles = np.linspace(0, 1, n_antenna, endpoint=False)*2*np.pi
        x = np.cos(angles)*R
        y = np.sin(angles)*R

        xx, zz = np.meshgrid(x, z)
        yy, _ = np.meshgrid(y, z)

        positions = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
        
        positions = antenna.position_elements(positions)
        
        normals = np.zeros_like(positions)
        
        normals[:,0] = -positions[:,0]/R
        normals[:,1] = -positions[:,1]/R
        
        #set polarization such that at position = (R, 0, 0) -> pol = (0, -1, 0)
        polarizations = np.zeros_like(positions)
        polarizations[:, 0] = positions[:,1]/R
        polarizations[:, 1] = -positions[:,0]/R

        instance = cls(positions, normals, polarizations, antenna)

        return instance

