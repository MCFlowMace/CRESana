

"""

Author: F. Thomas
Date: July 20, 2021

"""

__all__ = []

import numpy as np
from scipy.stats import norm

from .physicsconstants import speed_of_light
from .axialmotion import get_z, get_z_max, get_omega_axial
from .cyclotronmotion import get_avg_omega_cyclotron, get_slope

def get_signal(A, phase):

    return A*np.exp(1.0j*phase)

def get_cyclotron_phase(t, w_avg, w_m):

    return t*(w_avg - w_m)

def get_energy_loss_phase(t, slope):

    return 0.5*slope*t**2

def get_phase_shift(d, w_avg):

    return w_avg*d/speed_of_light

def get_distance(pos_e, pos_antenna):

    return np.sqrt(np.sum((pos_e-pos_antenna)**2, axis=1))

def get_fake_AM(x, sigma, A_0):

    if sigma==0.0:
        print('No AM')
        amplitude = A_0
    else:
        print('Use AM')
        #amplitude = A_0*norm.pdf(x, scale=sigma)
        amplitude = A_0 * np.exp(-0.5*(x/sigma)**2)

    return amplitude

class Sampler:

    def __init__(self, sr):

        self._sr = sr
        self._dt = 1/sr

    def _get_sample_times(self, N, t_0):

        return np.arange(N)*self._dt + t_0

    def __call__(self, N, t_0=0):

        return self._get_sample_times(N, t_0)

class SimpleElectron:

    def __init__(self, B, L_0, E_kin, theta):

        self._B = B
        self._L_0 = L_0
        self._E_kin = E_kin
        self._theta = theta
        self._z_max = get_z_max(self._theta, L_0)
        print('z_max', self._z_max)

    def traj(self):

        w_a = get_omega_axial(self._E_kin, self._theta, self._L_0)

        return lambda t: self.pos_vector(get_z(t, self._z_max, w_a, 0.0))


    def pos_vector(self, z):

        pos = np.zeros((z.shape[0], 3))
        pos[:,-1] = z

        return pos

    def get_w_cyclotron(self):

        return get_avg_omega_cyclotron(self._B, self._E_kin, self._z_max, self._L_0)

    def get_slope(self):

        return get_slope(self._E_kin, self._theta, self._B, self._z_max, self._L_0)

class SimpleSignal:

    def __init__(self, pos_antenna, w_mix, sr, sigma, energy_loss=False):

        self._sampler = Sampler(sr)
        self._pos = pos_antenna
        self._w_mix = w_mix
        self._sigma = sigma
        self._energy_loss = energy_loss

    def get_samples(self, N, electron):

        w_0 = electron.get_w_cyclotron()

        slope = electron.get_slope()

       # print('cyclotron frequency ', w_0/(2*np.pi*1e9))

        pos_e, d = self.distance(N, electron)

        phase = get_phase_shift(d, w_0)
        phase += get_cyclotron_phase(self._sampler(N), w_0, self._w_mix)

        if self._energy_loss:
            phase += get_energy_loss_phase(self._sampler(N), slope)

        relative_z = pos_e[:,-1] - self._pos[-1]
        A = get_fake_AM(relative_z, self._sigma, 1.0)#/d

        return get_signal(A, phase)

    def distance(self, N, electron):

        t = self._sampler(N)

        pos_e = electron.traj()(t)

        return pos_e, get_distance(pos_e, self._pos)





