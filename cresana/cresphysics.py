

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
    

def get_beta(E_kin):
    v_0 = get_relativistic_velocity(E_kin)
    return v_0/speed_of_light
    

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
    beta = get_beta(E_kin)
    w0 = get_omega_cyclotron(B, 0.0)
    scaling_factor = beta**2*np.sin(theta)**2/(1-beta**2)

    return w0**2/(6*np.pi*epsilon0*speed_of_light)*scaling_factor
    
    
def get_directive_gain(E_kin, pitch, angle):
    """
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.36.1498
    """
    def f_factor(beta_parallel, beta_ortho, theta):
    
        g_parallel = 1-beta_parallel*np.cos(theta)
        
        denominator = 4*(g_parallel**2-beta_ortho**2*np.sin(theta)**2)**(7/2)
        
        enum1 = 4*g_parallel**2*((1+beta_parallel**2)*(1+np.cos(theta)**2) - 4*beta_parallel*np.cos(theta))
        enum2 = (1- beta_parallel**2 + 3*beta_ortho**2)*beta_ortho**2*np.sin(theta)**4
        
        enum = enum1 - enum2
        
        return enum/denominator
    
    def g_factor(beta_parallel, beta_ortho, theta):
        
        beta = np.sqrt(beta_parallel**2 + beta_ortho**2)
        
        return 0.75*(1-beta**2)**2*f_factor(beta_parallel, beta_ortho, theta)
        
    def doppler_correction(g, beta_parallel, theta):
        return 1/(1 - beta_parallel*np.cos(theta))*g
        
    beta = get_beta(E_kin)
    beta_parallel = beta*np.cos(pitch)
    beta_ortho = beta*np.sin(pitch)
    g = g_factor(beta_parallel, beta_ortho, angle)
    return doppler_correction(g, beta_parallel, angle)
    

def get_slope(E_kin, p, w):
    return p*w/(E0_electron + E_kin)
