

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

from abc import ABC, abstractmethod

from scipy.signal import sawtooth
import numpy as np

from .physicsconstants import speed_of_light, E0_electron
from .utility import get_pos

class Trap(ABC):

    @abstractmethod
    def trajectory(self, electron):
        pass

    @abstractmethod
    def B_field(self, z):
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

        return lambda t: get_pos(   np.zeros(t.shape),
                                    np.zeros(t.shape),
                                    get_z_harmonic(t, z_max, omega, 0.0))

    def B_field(self, z):

        return harmonic_potential(z, self._B0, self._L0)

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

        return lambda t: get_pos(   np.zeros(t.shape),
                                    np.zeros(t.shape),
                                    get_z_flat(t, z_max, omega, 0.0))

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

        return lambda t: get_pos(   np.zeros(t.shape),
                            np.zeros(t.shape),
                            self._get_z(electron, t))

    def B_field(self, z):

        B = flat_potential(z, self._B0)

        left_harmonic = z < -self._L/2
        right_harmonic = z > self._L/2

        z_left_harmonic = z[left_harmonic] + self._L/2
        z_right_harmonic = z[right_harmonic] - self._L/2

        B[left_harmonic] = harmonic_potential(z_left_harmonic, self._B0, self._L0)
        B[right_harmonic] = harmonic_potential(z_right_harmonic, self._B0, self._L0)

        return B

    def _period(self, electron):

        flat_time = self._L/(electron.v0*np.cos(electron.pitch))
        harmonic_time = np.pi/get_omega_harmonic(electron.v0, electron.pitch, self._L0)

        return 2*(flat_time+harmonic_time)

    def _get_z(self, electron, t):

        v_axial = electron.v0 * np.cos(electron.pitch)
        omega = self._get_omega(electron)
        z_max = self._get_z_max(electron)
        T = self._period(electron)

        # z(t=0) = left end of flat region
        t1 = self._L/v_axial # electron reaches right end of flat region -> goes into harmonic region
        t2 = t1 + np.pi/omega # electron reaches right end of flat region again
        t3 = t2 + t1 # electron reaches left end of flat region again -> goes into harmonic region

        t = t + t1/2 #zero point shifted such that z(0) = 0
        t = t%T # z periodic with T

        z = np.zeros(t.shape)

        first_flat = t<=t1
        right_harmonic = (t>t1)&(t<=2)
        second_flat = (t>t2)&(t<=t3)
        left_harmonic = t>t3

        z[first_flat] = -self._L/2 + v_axial*t[first_flat]
        z[right_harmonic] = self._L/2 + z_max * np.sin(omega*(t[right_harmonic] - t1))
        z[second_flat] = self._L/2 - v_axial*(t[second_flat] - t2)
        z[left_harmonic] = -self._L/2 - z_max * np.sin(omega*(t[left_harmonic] - t3))

        return z

    def _get_omega(self, electron):

        return get_omega_harmonic(electron.v0, electron.pitch, self._L0)

    def _get_z_max(self, electron):

        return get_z_max_harmonic(self._L0, electron.pitch)

    def get_f(self, electron):
        return 1/self._period(electron)


class ArbitraryTrap(Trap):

    def __init__(self):
        pass

    def trajectory(self, electron):
        pass

    def B_field(self, z):
        pass
