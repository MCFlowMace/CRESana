

"""

Author: F. Thomas
Date: July 19, 2021

"""

__all__ = []

import numpy as np
from .physicsconstants import speed_of_light, E0_electron

def get_omega_cyclotron(B, E_kin):

    '''
    B - magnetic field value in T
    E_kin - electron kinetic energy in eV
    '''

    return speed_of_light*speed_of_light*B/(E0_electron + E_kin)

def get_avg_omega_cyclotron(B, E_kin, z_max, L_0):

    w0 = get_omega_cyclotron(B, E_kin)

    return w0*(1+ z_max**2/(2*L_0**2))
