

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
    

def get_radiated_power(E_kin, pitch, B):
    beta = get_beta(E_kin)
    w0 = get_omega_cyclotron(B, 0.0)
    scaling_factor = beta**2*np.sin(pitch)**2/(1-beta**2)

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


def g_harmonic_n(beta_parallel, beta_ortho, n, theta):
    
    beta = np.sqrt(beta_parallel**2 + beta_ortho**2)
    
    b = beta_ortho*np.sin(theta)/(1-beta_parallel*np.cos(theta))
    
    jn2 = (jv(n, n*b))**2
    jnd2 = (jvp(n, n*b))**2
    
    factor = 3*(1-beta**2)**2 * n**2/(beta_ortho**2 * ( 1 - beta_parallel*np.cos(theta))**3)
    
    bessel_sum = beta_ortho**2*jnd2 + ((np.cos(theta) - beta_parallel)/np.sin(theta))**2 * jn2
    
    return factor*bessel_sum


def g_receive(beta_parallel, beta_ortho, theta, g_f):
    
    g = g_f(beta_parallel, beta_ortho, theta)
    
    alpha = 1/(1 - beta_parallel*np.cos(theta))
    
    
    return alpha*g


def get_analytic_power(E_cres, pitch, B, d, n=None, free_space=True):
    
    beta_cres = get_relativistic_velocity(E_cres)/speed_of_light
    beta_cres_parallel = beta_cres*np.cos(pitch)
    beta_cres_ortho = beta_cres*np.sin(pitch)
    P0 = get_radiated_power(E_cres, pitch, B)*ev
    w = get_omega_cyclotron(B, E_cres)
    
    #print(P0, get_free_space_loss(w, d), P_free_space_loss)
    la = 2*np.pi*speed_of_light/w
    A_eff = la**2/(4*np.pi)
    
    if n==None:
        g_f = g_factor
        P_in = P0*get_free_space_loss(w, d)#*tf_gain
    else:
        g_f = lambda beta_parallel, beta_ortho, theta : g_harmonic_n(beta_parallel, beta_ortho, n, theta)
        P_in = P0*get_free_space_loss(w*n, d)#*tf_gain
        
    if not free_space:
        P_in = P0
    
    return A_eff, w, lambda theta: g_receive(beta_cres_parallel, beta_cres_ortho, theta, g_f)*P_in


def get_cres_frequencies(sr, n_harmonics, B, E_cres):
    
    f_cyclotron = get_omega_cyclotron(B, E_cres)/(2*np.pi)
    f_harmonics = np.arange(n_harmonics)*f_cyclotron

    f_aliased = calc_aliased_frequencies(f_harmonics, sr)
    
    return f_aliased


def get_cres_powers(E_cres, pitch, B, d, n_harmonics, polar):
    
    p_n = np.zeros(n_harmonics)

    for i in range(1, n_harmonics):
        _, _, p_f = get_analytic_power(E_cres, pitch, B, d, n=i, free_space=False)
        p_n[i] = p_f(polar)/(4*np.pi)
        
    return p_n


def get_cres_power_spec(E_cres, pitch, B, d, n_harmonics, polar, sr):
    
    f = get_cres_frequencies(sr, n_harmonics, B, E_cres)
    p = get_cres_powers(E_cres, pitch, B, d, n_harmonics, polar)
    
    f_hermitian = np.concatenate((-f[-1:0:-1], f))
    p_hermitian = np.concatenate((p[-1:0:-1]/2, p/2))
    
    return f_hermitian, p_hermitian
    
    
# polarization
    
def get_system(r):
    
    if np.all(np.abs(r - np.array([0,0,1])) <1e-7):
        return np.array([0,-1,0]), np.array([1,0,0])
    
    a = np.array([0,0,1])
    
    x = np.cross(r, a)
    x = x/np.sqrt(np.sum(x**2))
    y = np.cross(r, x)
    
    y = y/np.sqrt(np.sum(y**2))
    
    return x, y
    
    
def get_amplitudes(r, B_dir):
    
    costheta = np.dot(r_norm, B_dir)
    a_x = 1/np.sqrt(costheta**2 + 1)
    a_y = costheta*a_x
    
    return a_x, a_y
    
    
def get_polarization_vectors(r, B_dir):
    
    r_norm = r/np.sqrt(np.sum(r**2))
    
    x, y = get_system(r_norm)
    a_x, a_y = get_amplitudes(r_norm, B_dir)
    
    return a_x*x, a_y*y

#~ def get_slope(E_kin, p, w):
    #~ return p*w/(E0_electron + E_kin)


#~ def get_omega_cyclotron_time_dependent(B, E_kin, p, t):
    #~ w = get_omega_cyclotron(B, E_kin)

    #~ return w*(1+p*t/(E0_electron + E_kin))
    
    
