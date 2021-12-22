

"""

Author: F. Thomas
Date: July 19, 2021

"""

__all__ = []

import numpy as np
from .physicsconstants import speed_of_light, E0_electron, epsilon0

def get_relativistic_velocity(E_kin):

    '''
    E_kin - electron kinetic energy in eV
    '''

    relative_energy = E0_electron/(E0_electron + E_kin)

    return np.sqrt(1-relative_energy**2)*speed_of_light
    

def get_omega_cyclotron(B, E_kin):

    '''
    B - magnetic field value in T
    E_kin - electron kinetic energy in eV
    '''

    return speed_of_light*speed_of_light*B/(E0_electron + E_kin)
    

def get_omega_cyclotron_time_dependent(B, E_kin, p, t):

    w = get_omega_cyclotron(B, E_kin)

    return w*(1+p*t/(E0_electron + E_kin))
    

def get_radiated_power(E_kin, theta, B):

    v_0 = get_relativistic_velocity(E_kin)
    beta = v_0/speed_of_light
    w0 = get_omega_cyclotron(B, 0.0)
    scaling_factor = beta**2*np.sin(theta)**2/(1-beta**2)

    return w0**2/(6*np.pi*epsilon0*speed_of_light)*scaling_factor
    

def get_slope(E_kin, p, w):

    return p*w/(E0_electron + E_kin)
