

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
        
    def get_directivity_gain(self, angle_E_plane, angle_H_plane):
        return self.directivity_factor(angle_E_plane, angle_H_plane)**2
        
    def get_gain(self, angle_E_plane, angle_H_plane, w_receiver):
        g = self.get_directivity_gain(angle_E_plane, angle_H_plane)*self.get_tf_gain(w_receiver)
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
                    angle_E_plane, angle_H_plane, w_receiver, real_signal=False):
                        
        gain = self.get_gain(angle_E_plane, angle_H_plane, w_receiver)
        U0 = self.power_to_amplitude(copolar_power*gain)
        phase0 = self.get_phase(field_phase, w_receiver)
        element_signals = get_signal(U0, phase0, real=real_signal)
        
        return self.sum_elements(element_signals)
        
    @abstractmethod
    def transfer_function(self, w_receiver):
        pass
        
    @abstractmethod
    def directivity_factor(self, angle_E_plane, angle_H_plane):
        pass
        
    @abstractmethod
    def position_elements(self, positions):
        pass
        
    @abstractmethod
    def sum_elements(self, signals):
        pass
        

class IsotropicAntenna(Antenna):
    
    def __init__(self, gain=1.):
        Antenna.__init__(self)
        self.gain = gain
        
    def transfer_function(self, w_receiver):
        
        tf_abs = np.sqrt(standard_impedance*speed_of_light**2*np.pi*self.gain/(free_space_impedance*w_receiver**2))
        
        return tf_abs + 0.0j
        
    def directivity_factor(self, angle_E_plane, angle_H_plane):
        return np.ones_like(angle_E_plane)
        
    def position_elements(self, positions):
        return positions
        
    def sum_elements(self, signals):
        return signals


def get_dipole_factor(angle_E_plane, angle_H_plane):
    # beta is an angle in E_plane but 90deg at normal
    beta = np.pi/2 - angle_E_plane
    
    E_plane_factor = np.zeros_like(beta)
    
    nonzero = (beta != 0.0)&(np.abs(beta) != np.pi)
    
    E_plane_factor[nonzero] = np.cos(np.pi/2*np.cos(beta[nonzero]))/np.sin(beta[nonzero])
    H_plane_factor = np.cos(angle_H_plane)

    return E_plane_factor, H_plane_factor


class GenericAntenna(Antenna):
    
    def __init__(self, directivity_exponent, gain_dB):
        
        Antenna.__init__(self)
        self.directivity_exponent = directivity_exponent
        self.lin_gain =  10**(gain_dB/10)
        
    def transfer_function(self, w_receiver):
        
        tf_abs = np.sqrt(self.lin_gain*standard_impedance*speed_of_light**2*np.pi/(free_space_impedance*w_receiver**2))
        
        return tf_abs + 0.0j
        
    def directivity_factor(self, angle_E_plane, angle_H_plane):

        E_plane_factor, H_plane_factor = get_dipole_factor(angle_E_plane, angle_H_plane)
        
        return E_plane_factor**self.directivity_exponent*H_plane_factor
        
    def position_elements(self, positions):
        return positions
        
    def sum_elements(self, signals):
        return signals
    
class DipoleAntenna(GenericAntenna):
    def __init__(self, gain_dB):
        DipoleAntenna.__init__(self, directivity_exponent=1, gain_dB=gain_dB)
        
    def directivity_factor(self, angle_E_plane, angle_H_plane):
        # beta is an angle in E_plane but 90deg at normal
        beta = np.pi/2 - angle_E_plane
    
        E_plane_factor = np.zeros_like(beta)
    
        nonzero = (beta != 0.0)&(np.abs(beta) != np.pi)
    
        E_plane_factor[nonzero] = np.cos(np.pi/2*np.cos(beta[nonzero]))/np.sin(beta[nonzero])
        
        # H_plane_factor is 1 and thus does not show up explicitly in calculation
        return E_plane_factor

    
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
        
    def directivity_factor(self, angle_E_plane, angle_H_plane):
    
        E_plane_factor, H_plane_factor = get_dipole_factor(angle_E_plane, angle_H_plane)
        
        return E_plane_factor*H_plane_factor
        
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
        # position and orientation of antennas are defined by positions, normals and polarizations (E-plane), 
        # physical extent does not matter here.
        # normal direction of antennas
        # normal and polarization vector define E-plane (plane in which E-field oscillates)
        # normal and cross_polarization vector define H-plane (plane in which B-field oscillates)
        
        self.positions = positions
        self.normals = normals
        self.polarizations = polarizations
        self.cross_polarizations = normalize(np.cross(normals, polarizations))
        self.antenna = antenna
        
    def get_directivity_angles(self, d_vec):
        
        r_project_pol = project_on_plane(-d_vec, self.cross_polarizations)
        r_project_cross = project_on_plane(-d_vec, self.polarizations)
        
        # angle in H-plane relative to normal vector
        angle_H_plane = angle_with_orientation(np.expand_dims(self.normals, 1), r_project_pol, 
                                       np.expand_dims(self.cross_polarizations, 1))

        # angle in E-plane relative to normal vector
        angle_E_plane = angle_with_orientation(np.expand_dims(self.normals, 1), r_project_cross, 
                                       np.expand_dims(self.polarizations, 1))
        
        return angle_E_plane, angle_H_plane
        
    def get_polarization_mismatch(self, pol_x, pol_y, delta_phase):
    
        a = np.einsum('ik,ijk->ij', self.polarizations, pol_x) #(dot(pol_x, polarizations))
        b = np.einsum('ik,ijk->ij', self.polarizations, pol_y)
        ab = 2*np.cos(delta_phase)*a*b
        
        return a**2 + b**2 + ab
        
    def get_received_copolar_field_power(self, P_transmitted, w_transmitter, pol_x, pol_y, d):
        
        polarization_mismatch_gain = self.get_polarization_mismatch(pol_x, pol_y, np.pi/2)
        
        P_received = calculate_received_power(P_transmitted, w_transmitter, 
                                        polarization_mismatch_gain, d**2)
        
        return P_received
        
    def get_voltage(self, received_copolar_field_power, field_phase, 
                    w_receiver, d_vec, real_signal=False):
                        
        angle_E_plane, angle_H_plane = self.get_directivity_angles(d_vec)
        
        return self.antenna.get_voltage(received_copolar_field_power, 
                                        field_phase, angle_E_plane, angle_H_plane,
                                        w_receiver, real_signal=real_signal)
        
    @classmethod
    def make_multi_ring_array(cls, R, n_antenna, n_rings, z_min, z_max, 
                                antenna, add_orthogonal_polarizations=False):

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
        
        if add_orthogonal_polarizations:
            polarizations[1::2, 0] = -positions[1::2,0]/R
            polarizations[1::2, 1] = positions[1::2,1]/R

        instance = cls(positions, normals, polarizations, antenna)

        return instance

    @classmethod
    def make_generic_full_cylinder_array(cls, L, R, w, antenna, down_scale_L=1, down_scale_R=1, add_orthogonal_polarizations=False):

        la = 2*np.pi*speed_of_light/w
        gain = antenna.get_tf_gain(w) #this will not work correctly for a slotted waveguide antenna

        n_rings = int(L/la*np.sqrt(4*np.pi/gain))//down_scale_L
        n_antenna = int(2*np.pi*R/la*np.sqrt(4*np.pi/gain))//down_scale_R

        print(f'Using array with {n_rings} rings and {n_antenna} antennas per ring')

        return cls.make_multi_ring_array(R, n_antenna, n_rings, -L/2, L/2, antenna, add_orthogonal_polarizations)

