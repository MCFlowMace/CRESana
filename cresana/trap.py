

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

from abc import ABC, abstractmethod

from scipy.signal import sawtooth, square
from scipy.optimize import root_scalar
from scipy.integrate import cumtrapz
from scipy.interpolate import CubicSpline
import numpy as np

from .physicsconstants import speed_of_light, E0_electron
from .utility import get_pos
from .electronsim import ElectronSim


def magnetic_moment(E_kin, pitch, B0):
    return E_kin * np.sin(pitch)**2/B0
    

class Trap(ABC):

    @abstractmethod
    def trajectory(self, electron):
        pass

    @abstractmethod
    def B_field(self, z):
        pass
    
    @abstractmethod
    def pitch(self, electron):
        pass
        
    @abstractmethod
    def get_f(self, electron):
        pass
        

def harmonic_potential(z, B0, L0):
    return B0 * ( 1 + z**2/L0**2)
    

def flat_potential(z, B0):
    if type(z) == np.ndarray:
        potential = np.full(z.shape, B0)
    else:
        potential = np.array([B0])

    return potential
    

def get_z_harmonic(t, z_max, omega, phi):
    return z_max*np.sin(omega*t + phi)
        
        
def get_vz_harmonic(t, z_max, omega, phi):
    return z_max*omega*np.cos(omega*t + phi)
    

def get_z_flat(t, z_max, omega, phi):
    return z_max*sawtooth(t*omega + np.pi/2 + phi, width=0.5)
        

def get_omega_harmonic(v0, pitch, L0):
    return v0*np.sin(pitch)/L0
    

def get_z_max_harmonic(L0, pitch):
    return L0/np.tan(pitch)
    

class HarmonicTrap(Trap):

    def __init__(self, B0, L0):
        self._B0 = B0
        self._L0 = L0

    def trajectory(self, electron):
        omega = self._get_omega(electron)
        z_max = self._get_z_max(electron)

        phi0 = np.arcsin(electron._z0/z_max)

        return lambda t: get_pos(   np.ones_like(t)*electron._x0,
                                    np.ones_like(t)*electron._y0,
                                    get_z_harmonic(t, z_max, omega, phi0))
                                    
    def B_field(self, z):
        return harmonic_potential(z, self._B0, self._L0)
        
    def pitch(self, electron):
        omega = self._get_omega(electron)
        z_max = self._get_z_max(electron)
        phi0 = np.arcsin(electron._z0/z_max)
        
        def f(t):
            vz = get_vz_harmonic(t, z_max, omega, phi0)
            return np.arccos(vz/electron.v0)
        
        return f
        
    def _get_omega(self, electron):
        return get_omega_harmonic(electron.v0, electron.pitch, self._L0)
        
    def _get_z_max(self, electron):
        return get_z_max_harmonic(self._L0, electron.pitch)
        
    def get_f(self, electron):
        return self._get_omega(electron)/(2*np.pi)
        

class BoxTrap(Trap):

    def __init__(self, B0, L):
        self._B0 = B0
        self._L = L

    def trajectory(self, electron):
        omega = self._get_omega(electron)
        z_max = self._get_z_max()
        phi0 = electron._z0/z_max*np.pi/2

        return lambda t: get_pos(   np.ones_like(t)*electron._x0,
                                    np.ones_like(t)*electron._y0,
                                    get_z_flat(t, z_max, omega, phi0))

    def pitch(self, electron):
        omega = self._get_omega(electron)
        z_max = self._get_z_max()
        phi0 =  electron._z0/z_max*np.pi/2
        
        def f(t):
            delta = np.pi/2 - electron.pitch
            sign = square(t*omega + np.pi/2 + phi0)
            return np.pi/2 - sign*delta
        
        return f
        
    def B_field(self, z):
        B = flat_potential(z, self._B0)

        B[z>self._L/2] = np.inf
        B[z<-self._L/2] = np.inf

        return B

    def _get_omega(self, electron):
        return electron.v0*np.cos(electron.pitch)*np.pi/self._L

    def _get_z_max(self):
        return self._L/2

    def get_f(self, electron):
        return self._get_omega(electron)/(2*np.pi)


class BathtubTrap(Trap):

    def __init__(self, B0, L, L0):
        self._B0 = B0
        self._L = L
        self._L0 = L0

    def trajectory(self, electron):
        return lambda t: get_pos(   np.ones_like(t)*electron._x0,
                                    np.ones_like(t)*electron._y0,
                                    self._get_z(electron, t))

    def B_field(self, z):
        # in case float input is used
        z_np = np.expand_dims(z, 0)

        B = flat_potential(z_np, self._B0)

        left_harmonic = z_np < -self._L/2
        right_harmonic = z_np > self._L/2

        z_left_harmonic = z_np[left_harmonic] + self._L/2
        z_right_harmonic = z_np[right_harmonic] - self._L/2

        B[left_harmonic] = harmonic_potential(z_left_harmonic, self._B0, self._L0)
        B[right_harmonic] = harmonic_potential(z_right_harmonic, self._B0, self._L0)

        return B[0] #undo the expand_dims in first line

    def _period(self, electron):
        flat_time = self._L/(electron.v0*np.cos(electron.pitch))
        harmonic_time = np.pi/get_omega_harmonic(electron.v0, electron.pitch, self._L0)

        return 2*(flat_time+harmonic_time)

    def _get_z(self, electron, t):
        v_axial = electron.v0 * np.cos(electron.pitch)
        omega = self._get_omega(electron)
        z_max = self._get_z_max(electron)
        
        if abs(electron._z0)>(z_max+self._L/2):
            raise ValueError(f'Electron cannot be trapped at z0={electron._z0} because for pitch={electron._pitch/np.pi*180}° z_max=+-{z_max+self._L/2:.3f}')
        
        T = self._period(electron)

        # z(t=0) = left end of flat region
        t1 = self._L/v_axial # electron reaches right end of flat region -> goes into harmonic region
        t2 = t1 + np.pi/omega # electron reaches right end of flat region again
        t3 = t2 + t1 # electron reaches left end of flat region again -> goes into harmonic region
        
        if abs(electron._z0) < self._L/2:
            t0 = electron._z0/v_axial
        elif electron._z0>0:
            t0 = t1/2 + 1/omega*np.arcsin((electron._z0 - self._L/2)/z_max)
        else:
            t0 = -t1/2 - 1/omega*np.arcsin((-electron._z0 - self._L/2)/z_max)

        t = t + t1/2 + t0 #zero point shifted such that z(0) = z0
        t = t%T # z periodic with T

        first_flat = t<=t1
        right_harmonic = (t>t1)&(t<=2)
        second_flat = (t>t2)&(t<=t3)
        left_harmonic = t>t3
        
        z = np.zeros(t.shape)
        z[first_flat] = -self._L/2 + v_axial*t[first_flat]
        z[right_harmonic] = self._L/2 + z_max * np.sin(omega*(t[right_harmonic] - t1))
        z[second_flat] = self._L/2 - v_axial*(t[second_flat] - t2)
        z[left_harmonic] = -self._L/2 - z_max * np.sin(omega*(t[left_harmonic] - t3))

        return z
        
    def pitch(self, electron):
        v_axial = electron.v0 * np.cos(electron.pitch)
        omega = self._get_omega(electron)
        z_max = self._get_z_max(electron)
        
        if abs(electron._z0)>(z_max+self._L/2):
            raise ValueError(f'Electron cannot be trapped at z0={electron._z0} because for pitch={electron._pitch/np.pi*180}° z_max=+-{z_max+self._L/2:.3f}')
        
        T = self._period(electron)

        # z(t=0) = left end of flat region
        t1 = self._L/v_axial # electron reaches right end of flat region -> goes into harmonic region
        t2 = t1 + np.pi/omega # electron reaches right end of flat region again
        t3 = t2 + t1 # electron reaches left end of flat region again -> goes into harmonic region
        
        if abs(electron._z0) < self._L/2:
            t0 = electron._z0/v_axial
        elif electron._z0>0:
            t0 = t1/2 + 1/omega*np.arcsin((electron._z0 - self._L/2)/z_max)
        else:
            t0 = -t1/2 - 1/omega*np.arcsin((-electron._z0 - self._L/2)/z_max)
            
        def f(t):

            t = t + t1/2 + t0 #zero point shifted such that z(0) = z0
            t = t%T # z periodic with T

            first_flat = t<=t1
            right_harmonic = (t>t1)&(t<=2)
            second_flat = (t>t2)&(t<=t3)
            left_harmonic = t>t3
            
            delta = np.pi/2 - electron.pitch
            
            pitch = np.zeros(t.shape)
            pitch[first_flat] = np.pi/2 - delta
            
            vz = get_vz_harmonic(t[right_harmonic] - t1, z_max, omega, 0.)
            pitch[right_harmonic] = np.arccos(vz/electron.v0) # self._L/2 + z_max * np.sin(omega*(t[right_harmonic] - t1))
            
            pitch[second_flat] = np.pi/2 + delta
            
            vz = -get_vz_harmonic(t[left_harmonic] - t3, z_max, omega, 0.)
            pitch[left_harmonic] = np.arccos(vz/electron.v0)
            #pitch[left_harmonic] = -self._L/2 - z_max * np.sin(omega*(t[left_harmonic] - t3))

            return pitch
            
        return f

    def _get_omega(self, electron):
        return get_omega_harmonic(electron.v0, electron.pitch, self._L0)

    def _get_z_max(self, electron):
        return get_z_max_harmonic(self._L0, electron.pitch)

    def get_f(self, electron):
        return 1/self._period(electron)


class ArbitraryTrap(Trap):

    def __init__(self, f, B0):
        self._profile = f
        self._B0 = B0

    def trajectory(self, electron):
        _, _, z_f = self._solve_trajectory(electron)

        return lambda t: get_pos(   np.zeros(t.shape),
                            np.zeros(t.shape),
                            z_f(t))

    def B_field(self, z):
        return self._profile(z)

    def get_f(self, electron):
        t, _, _ = self._solve_trajectory(electron)
        t_max = t[-1] - t[1]

        return 1/(2*t_max)

    def _solve_trajectory(self, electron):
        z_root_guess = 1
        E_kin = electron.E_kin
        mu = magnetic_moment(E_kin, electron.pitch, self._B0)

        def potential_difference(z):
            return E_kin - mu*self.B_field(z)

        left = root_scalar(potential_difference, method='secant', x0=-z_root_guess, x1=0.0)
        right = root_scalar(potential_difference, method='secant', x0=z_root_guess, x1=0.0)

        z = np.linspace(left.root+5e-14, right.root-5e-14, 100000)
        dz = z[1] - z[0]

        integrand = 1/np.sqrt(potential_difference(z))

        integral = cumtrapz(integrand, x=z, initial=0.0)

        t = integral * np.sqrt(E0_electron/2)/speed_of_light

        t_mod = t[1:-1] - t[1]

        interpolation = CubicSpline(t_mod, z[1:-1], bc_type='clamped')

        def z_f(t_in):

            t_out = t_in.copy()

            t_out += t_mod[-1]/2
            t_out %= 2*t_mod[-1]
            t_out[t_out>t_mod[-1]] = 2*t_mod[-1] - t_out[t_out>t_mod[-1]]

            return interpolation(t_out)

        return t, z, z_f
