

"""

Author: F. Thomas
Date: July 19, 2021

"""

__all__ = []

import numpy as np

def get_omega_axial(v_0, theta_bot, L_0):

    return v_0*np.sin(theta_bot)/L_0

def get_z_max(theta_bot, L_0):

    return L_0/np.tan(theta_bot)

def get_z(t, z_max, omega_a, phi_a):

    return z_max*np.sin(omega_a*t + phi_a)

