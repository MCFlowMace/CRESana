

"""

Author: F. Thomas
Date: July 19, 2021

"""

__all__ = []

import numpy as np

from .physicsconstants import speed_of_light, E0_electron, epsilon0
from .electronsim import ElectronSim
from .utility import norm_squared
from .physicsconstants import ev

from scipy.special import jv, jvp

#Frequently used user functions

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
    

def get_slope(E_kin, p, w):
    return p*w/(E0_electron + E_kin)


def get_omega_cyclotron_time_dependent(B, E_kin, p, t):
    w = get_omega_cyclotron(B, E_kin)

    return w*(1+p*t/(E0_electron + E_kin))


#Other functions mostly for internal usage

def _gain_total(beta_parallel, beta_ortho, theta):
    
    #https://journals.aps.org/pra/abstract/10.1103/PhysRevA.36.1498
    
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
        
    return g_factor(beta_parallel, beta_ortho, theta)


def _gain_harmonic_n(beta_parallel, beta_ortho, n, theta):
    
     #https://journals.aps.org/pra/abstract/10.1103/PhysRevA.36.1498
    
    beta = np.sqrt(beta_parallel**2 + beta_ortho**2)
    
    b = beta_ortho*np.sin(theta)/(1-beta_parallel*np.cos(theta))
    
    jn2 = (jv(n, n*b))**2
    jnd2 = (jvp(n, n*b))**2
    
    factor = 3*(1-beta**2)**2 * n**2/(beta_ortho**2 * ( 1 - beta_parallel*np.cos(theta))**3)
    
    #this produces nans for theta=0 because it tries to evaluate j_n(n*x)/x
    #can be solved in theory because j_1(x)/x -> 0.5 for x->0
    #and j_{n>1}(x)/x -> 0 for x->0
    #let's see if that case even shows up in actual sims
    bessel_sum = beta_ortho**2*jnd2 + ((np.cos(theta) - beta_parallel)/np.sin(theta))**2 * jn2
    
    return factor*bessel_sum


def _doppler_factor(beta_parallel, theta):
    
    alpha = 1/(1 - beta_parallel*np.cos(theta))
    
    return alpha
    

def _get_analytic_power(E_kin, pitch, B, theta, n=None):
    
    beta = get_beta(E_kin)
    beta_parallel = beta*np.cos(pitch)
    beta_ortho = beta*np.sin(pitch)
    P0 = get_radiated_power(E_kin, pitch, B)*ev
    gain_doppler = _doppler_factor(beta_parallel, theta)
    
    if n==None:
        gain = _gain_total(beta_parallel, beta_ortho, theta)
    else:
        gain = _gain_harmonic_n(beta_parallel, beta_ortho, n, theta)
        
    P_emitter = P0*gain_doppler*gain
        
    return P_emitter


def _get_cres_frequencies(sr, n_harmonics, B, E_cres):
    
    from .utility import calc_aliased_frequencies
    
    f_cyclotron = get_omega_cyclotron(B, E_cres)/(2*np.pi)
    f_harmonics = np.arange(n_harmonics)*f_cyclotron

    f_aliased = calc_aliased_frequencies(f_harmonics, sr)
    
    return f_aliased


def _get_cres_powers(E_kin, pitch, B, n_harmonics, theta):
    
    p_n = np.zeros(n_harmonics)

    for i in range(1, n_harmonics):
        p_n[i] = _get_analytic_power(E_kin, pitch, B, theta, n=i)
        
    return p_n


def get_cres_power_spec(E_cres, pitch, B, n_harmonics, theta, sr):
    
    f = _get_cres_frequencies(sr, n_harmonics, B, E_cres)
    p = _get_cres_powers(E_cres, pitch, B, n_harmonics, theta)
    
    f_hermitian = np.concatenate((-f[-1:0:-1], f))
    p_hermitian = np.concatenate((p[-1:0:-1]/2, p/2))
    
    return f_hermitian, p_hermitian
    
    
# polarization
    
def _get_polarization_vectors(r_norm, theta):
    
    def _get_right_handed_coordinate_system(r):
    
        a = np.array([0,0,1])
        
        if np.all(np.abs(r - a) <1e-7):
            return np.array([0,-1,0]), np.array([1,0,0])
        
        x = np.cross(r, a)
        x = x/np.sqrt(np.sum(x**2))
        y = np.cross(r, x)
        
        y = y/np.sqrt(np.sum(y**2))
        
        return x, y
        
    def _get_amplitudes(r, theta):
        
        costheta = np.cos(theta) #np.dot(r_norm, B_dir)
        a_x = 1/np.sqrt(costheta**2 + 1)
        a_y = costheta*a_x
        
        return a_x, a_y
    
    #r_norm = r/np.sqrt(np.sum(r**2))
    
    x, y = get_right_handed_coordinate_system(r_norm)
    a_x, a_y = get_amplitudes(r_norm, B_dir)
    
    return a_x*x, a_y*y
    

#Class to bundle all necessary analytic descriptions of the cyclotron fields
class AnalyticCyclotronField:
    
    def __init__(self, electronsim, n_harmonic=None):
        
        self.electronsim = electronsim
        # frequency is independent of point
        self.w = get_omega_cyclotron(electronsim.B_vals, electronsim.E_kin)
        self.n_harmonic = n_harmonic
        
    @classmethod
    def make_from_params(cls, E_kin, pitch, B, n_harmonic=None):
        
        coords = np.array([0., 0., 0.])
        t = np.array([0.])
        B_vals = np.array(np.sqrt(norm_squared(np.expand_dims(B,0))))
        pitch_vals = np.array([pitch])
        
        electronsim = ElectronSim(coords, t, B_vals, E_kin, pitch_vals, B/B_vals[0])
        instance = cls(electronsim, n_harmonic)
        
        return instance
        
    def calc_d_vec_and_abs(self, pos):
        r = np.expand_dims(pos, 1) - self.electronsim.coords
        
        d = np.sqrt(norm_squared(r))
        d_vec = r/d
        
        return d_vec, d
        
    def calc_polar_angle(self, r_norm):

        return np.arccos(np.dot(r_norm, self.electronsim.B_dir))
        
    def get_field_parameters(self, p):
        """
        Get the parameters that analytically describe the field at point p.
        
        Parameters
        ---------
        p : 2D np.array
            points to evaluate the fields at
            
        Returns
        ---------
        w : 1D np.array
            cyclotron frequency for each time sample of the electron simulation
        P : 2D np.array
            emitted power for the 1st harmonic of the field spectrum in the 
            direction of each point in p and for each time sample of the
            electron simulation
        pol_x : 3D np.array
            x vectors of polarization state. First axis is time, second
            axis are the points p and third axis are the coordinates
        pol_y : 3D np.array
            y vectors of polarization state. First axis is time, second
            axis are the points p and third axis are the coordinates
        """
        
        d_vec, d = self.calc_d_vec_and_abs(pos)
        theta = self.calc_polar_angle(d_vec)
        
        power = _get_analytic_power(self.electronsim.E_kin, 
                                self.electronsim.pitch, 
                                self.electronsim.B_vals, 
                                theta, self.n_harmonic)
        
        pol_x, pol_y = _get_polarization_vectors(d_vec, theta)
        
        return self.w, power, pol_x, pol_y, d_vec, d, theta
