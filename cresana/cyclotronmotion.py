

"""

Author: F. Thomas
Date: July 19, 2021

"""

__all__ = []

import numpy as np
from .physicsconstants import speed_of_light, E0_electron, epsilon0
from .axialmotion import get_relativistic_velocity

def get_omega_cyclotron(B, E_kin):

    '''
    B - magnetic field value in T
    E_kin - electron kinetic energy in eV
    '''

    return speed_of_light*speed_of_light*B/(E0_electron + E_kin)

def get_avg_omega_cyclotron(B, E_kin, z_max, L_0):

    w0 = get_omega_cyclotron(B, E_kin)

    return w0*(1+ z_max**2/(2*L_0**2))

# ~ def get_radiated_power(E_kin, theta, B, z_max, L_0):

    # ~ v_0 = get_relativistic_velocity(E_kin)
    # ~ beta = v_0/speed_of_light

    # ~ scaling_factor = beta**2*np.sin(theta)**2/(1-beta**2)

    # ~ w_0 = get_avg_omega_cyclotron(B, E_kin, z_max, L_0)

    # ~ return w_0**2/(6*np.pi*epsilon0*speed_of_light)*scaling_factor

# ~ def get_slope(E_kin, theta, B, z_max, L_0):

    # ~ p = get_radiated_power(E_kin, theta, B, z_max, L_0)
    # ~ w_0 = get_avg_omega_cyclotron(B, E_kin, z_max, L_0)

    # ~ return p*w_0/(E0_electron + E_kin)
    
def get_radiated_power(E_kin, theta, B, w):

    v_0 = get_relativistic_velocity(E_kin)
    beta = v_0/speed_of_light

    scaling_factor = beta**2*np.sin(theta)**2/(1-beta**2)

    return w**2/(6*np.pi*epsilon0*speed_of_light)*scaling_factor

def get_slope(E_kin, p, w):

    return p*w/(E0_electron + E_kin)
