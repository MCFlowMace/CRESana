

"""

Author: F. Thomas
Date: July 26, 2021

"""

__all__ = []

from abc import ABC, abstractmethod

from scipy.signal import sawtooth

def get_relativistic_velocity(E_kin):

    '''
    E_kin - electron kinetic energy in eV
    '''

    relative_energy = E0_electron/(E0_electron + E_kin)

    return np.sqrt(1-relative_energy**2)*speed_of_light

def get_pos(x, y, z):

    return np.vstack((x,y,z)).transpose()

def get_x(R, phi):

    return R*np.cos(phi)

def get_y(R, phi):

    return R*np.sin(phi)

def gradB_phase(t, omega, phi):

    return t*omega + phi

class Electron:

    def __init__(self, E_kin, pitch, r=0, phi=0, z0=0, v_phi=0):

        self._E_kin = E_kin
        self._pitch = pitch/180*np.pi
        self._x0 = r*np.cos(phi)
        self._y0 = r*np.sin(phi)
        self._z0 = z0
        self._v_phi = v_phi
        self._v0 = get_relativistic_velocity(E_kin)

    @property
    def E_kin:
        return self._E_kin

    @property
    def pitch:
        return self._pitch

    @property
    def v0:
        return self._v0


class Trap(ABC):

    @abstractmethod
    def trajectory(self):
        pass

    @abstractmethod
    def B_field(self, z):
        pass

class HarmonicTrap(Trap):

    def __init__(self, B0, L0):

        self._B0 = B0
        self._L0 = L0

    def trajectory(self, electron: Electron):

        omega = self._get_omega(electron)
        z_max = self._get_z_max(electron)

        return lambda t: get_pos(   np.zeros(t.shape),
                                    np.zeros(t.shape),
                                    self._get_z(t, z_max, omega, 0.0))

    def B_field(self, z):

        return self._B0 * ( 1 + z**2/self._L0**2)

    def _get_omega(self, electron: Electron):

        return electron.v0*np.sin(electron.pitch)/self._L0

    def _get_z_max(self, electron: Electron):

        return self._L0/np.tan(electron.pitch)

    def _get_z(t, z_max, omega, phi):

        return z_max*np.sin(omega*t + phi)

class BoxTrap(Trap):

    def __init__(self, B0, L):

        self._B0 = B0
        self._L = L

    def trajectory(self, electron: Electron):

        omega = self._get_omega(electron)
        z_max = self._get_z_max()

        return lambda t: get_pos(   np.zeros(t.shape),
                                    np.zeros(t.shape),
                                    self._get_z(t, z_max, omega, 0.0))

    def B_field(self, z):

        B = np.full(z.shape, self._B0)

        B[z>self._L/2] = np.inf
        B[z<-self._L/2] = np.inf

        return B

    def _get_omega(self, electron: Electron):

        return electron.v0*np.cos(electron.pitch)*np.pi/self._L

    def _get_z_max(self):

        return self._L/2

    def _get_z(t, z_max, omega, phi):

        return z_max*sawtooth(t*omega + np.pi/2 + phi, width=0.5)

class BathtubTrap(Trap):

    def __init__(self):
        pass

    def trajectory(self, electron: Electron):
        pass

    def B_field(self, z):
        pass

class ArbitraryTrap(Trap):

    def __init__(self):
        pass

    def trajectory(self, electron: Electron):
        pass

    def B_field(self, z):
        pass
