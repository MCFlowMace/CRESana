

"""

Author: F. Thomas
Date: July 26, 2021

"""

__all__ = []

from abc import ABC, abstractmethod

from scipy.signal import sawtooth
import numpy as np

from .physicsconstants import speed_of_light, E0_electron

def get_relativistic_velocity(E_kin):

    '''
    E_kin - electron kinetic energy in eV
    '''

    relative_energy = E0_electron/(E0_electron + E_kin)

    return np.sqrt(1-relative_energy**2)*speed_of_light

def get_x(R, phi):

    return R*np.cos(phi)

def get_y(R, phi):

    return R*np.sin(phi)

def gradB_phase(t, omega, phi):

    return t*omega + phi

class Electron:

    """Represents the initial state of an electron.

    Attributes
    ----------
    E_kin : float
        Initial kinetic energy.
    pitch : float
        Initial pitch angle at trap minimum.
    v0    : float
        Initial velocity.
    """

    def __init__(self, E_kin, pitch, r=0, phi=0, z0=0, v_phi=0):

        self._E_kin = E_kin
        self._pitch = pitch/180*np.pi
        self._x0 = r*np.cos(phi)
        self._y0 = r*np.sin(phi)
        self._z0 = z0
        self._v_phi = v_phi
        self._v0 = get_relativistic_velocity(E_kin)

    @property
    def E_kin(self):
        return self._E_kin

    @property
    def pitch(self):
        return self._pitch

    @property
    def v0(self):
        return self._v0

    def __repr__(self):

        return ("Kinetic energy : {0:10.4f} eV \n".format(self._E_kin)
                +"Pitch angle:  {0:8.4f} ° \n".format(self._pitch/np.pi*180)
                +"Velocity:  {0:14.2f} m/s \n".format(self._v0)
                +"X0:  {0:6.4f} m \n".format(self._x0)
                +"Y0:  {0:6.4f} m \n".format(self._y0)
                +"Z0:  {0:6.4f} m \n".format(self._z0)
                +"Velocity angle:  {0:6.4f} \n".format(self._v_phi)
                )


class ElectronSim:

    """Represents an electron simulation result.

    Attributes
    ----------
    coords : numpy.ndarray
        Trajectory of the electron.
    t      : numpy.ndarray
        Sampling times.
    B_vals : numpy.ndarray
        Absolute B-field experienced by the electron.
    """

    def __init__(self, coords, t, B_vals):

        self.coords = coords
        self.t = t
        self.B_vals = B_vals


def simulate_electron(electron, sampler, trap):

    t = sampler()
    coords = trap.trajectory(electron)(t)
    B_vals = trap.B_field(coords[:,2])

    return ElectronSim(coords, t, B_vals)
