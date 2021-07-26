

"""

Author: F. Thomas
Date: July 19, 2021

"""

__all__ = []

import numpy as np
from scipy.signal import sawtooth

from .physicsconstants import speed_of_light, E0_electron

def get_omega_axial(E_kin, theta_bot, L_0):

    v_0 = get_relativistic_velocity(E_kin)
    print(v_0)
    return v_0*np.sin(theta_bot)/L_0
    #z_max = get_z_max(theta_bot, L_0)
   # return v_0/z_max

def get_z_max(theta_bot, L_0):

    return L_0/np.tan(theta_bot)

def get_z(t, z_max, omega_a, phi_a):

    return z_max*np.sin(omega_a*t + phi_a)

def get_relativistic_velocity(E_kin):

    '''
    E_kin - electron kinetic energy in eV
    '''

    relative_energy = E0_electron/(E0_electron + E_kin)

    return np.sqrt(1-relative_energy**2)*speed_of_light

def get_z_flat(t, z_max, omega_a, phi_a):

    return z_max*sawtooth(t*omega_a + np.pi/2 + phi_a, width=0.5)

def get_omega_flat(E_kin, theta_bot, L):

    v_0 = get_relativistic_velocity(E_kin)

    return v_0*np.cos(theta_bot)*np.pi/L
