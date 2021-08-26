

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

from abc import ABC, abstractmethod
from scipy.interpolate import RectBivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def fake_AM(x, sigma, A_0):
    return A_0 * np.exp(-0.5*(x/sigma)**2)

def get_disc_solid_angle(d, R, r):

    """
    d - distance between source and disc
    R - radius of the disc
    r - radial position of the source
    """

    b = np.broadcast(d, r)
    res = np.empty(b.shape)
    d_broad = np.broadcast_to(d, b.shape)
    r_broad = np.broadcast_to(r, b.shape)

    ind = r_broad/R>0.4 #maximum error < 10%

    res[ind] = get_disc_solid_angle_general(d_broad[ind], r_broad[ind], R)
    res[~ind] = get_disc_solid_angle_on_axis(d_broad[~ind], R)

    return res

def get_disc_solid_angle_on_axis(d, R):

    """
    d - distance between source and disc
    R - radius of the disc
    """
    r_b = np.sqrt(d**2 + R**2)
    h = r_b - d

    return np.pi*2*h/r_b

def get_disc_solid_angle_general(d, r, R):

    """
    d - distance between source and disc
    R - radius of the disc
    r - radial position of the source
    """

    # reference: https://aip.scitation.org/doi/pdf/10.1063/1.1716590

    def K(k):
        return special.ellipk(k**2)

    def AlphaSqr(r_0,r_m):
        return 4*np.abs(r_0)*np.abs(r_m)/(np.abs(r_0)+np.abs(r_m))**2

    def R1(L,r_0,r_m):
        return np.sqrt(L**2+(np.abs(r_m)-np.abs(r_0))**2)

    def RMax(L,r_0,r_m):
        return np.sqrt(L**2+(np.abs(r_m)+np.abs(r_0))**2)

    def fk(L,r_0,r_m):
        R_1=R1(L,r_0,r_m)
        R_Max=RMax(L,r_0,r_m)
        return np.sqrt(1-(R_1/R_Max)**2)

    def Xi(L,r_0,r_m):
        return np.arctan2(L,np.abs(r_0-r_m))

    def KM1(k):
        return special.ellipkm1(k**2)

    #logic from here https://math.stackexchange.com/q/629326
    def heuman_lambda(xi,k):
        k_bar=np.sqrt(1-k**2)
        E_k= special.ellipe(k**2)
        F_xik= special.ellipkinc(xi,k_bar**2)
        K_k=K(k)
        E_xik=special.ellipeinc(xi, k_bar**2)
        return 2*(E_k*F_xik+K_k*E_xik-K_k*F_xik)/np.pi

    R_max=RMax(d, r, R)
    R_1=R1(d, r, R)
    k=fk(d, r, R)
    alpha_sqr=AlphaSqr(r, R)
    xi=Xi(d, r, R)

    return 2*np.pi-(2*d/R_max)*K(k)-np.pi*heuman_lambda(xi, k)

class FileGainPattern:

    """
    Author: R. Reimann
    """

    def __init__(self, data_paths):
        self.load_power_map(data_paths)
        self.clean_spikes()
        self.generate_spline()

    def load_power_map(self, data_paths):
        power = None
        radius = None
        z_pos = None
        for path in data_paths:
            data = np.genfromtxt(path, delimiter=",")

            z = data.flatten()[::4]/1000 #in m
            r = data.flatten()[1::4]/1000 #in m
            p = data.flatten()[2::4]

            if power is None:
                z_pos = np.array(sorted(set(z)))
                radius = np.array(sorted(set(r)))
                power = p.reshape((len(set(z)), len(set(r))))
            else:
                if not all(z_pos == np.array(sorted(set(z)))):
                    print("Warning")
                radius = np.concatenate([radius, np.array(sorted(set(r)))])
                power = np.concatenate([power, p.reshape((len(set(z)), len(set(r))))], axis=-1)
        self.r_pos = z_pos
        self.d_pos = radius
        self.power = power#/np.max(power)

    def clean_spikes(self):
        for i in range(1, len(self.power)-1):
            for j in range(1, len(self.power[i])-1):
                mean = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di==dj==0: continue
                        mean.append(self.power[i+di,j+dj])
                if np.abs((np.mean(mean)-self.power[i,j])/np.mean(mean)) > 0.7:
                    self.power[i,j] = np.mean(mean)

    def generate_spline(self):
        self.spline = RectBivariateSpline(self.r_pos, self.d_pos, self.power)

    def __call__(self, r, d, grid=False):
        return self.spline(r, d, grid=grid)

    def plot(self, **kwargs):
        fig, ax = plt.subplots(figsize=(3*2,6*2))

        extent = [np.min(self.r_pos), np.max(self.r_pos), np.min(self.d_pos), np.max(self.d_pos)]

        power_map = self.spline(self.r_pos, self.d_pos, grid=True)
        print(power_map.shape)
        im = ax.imshow(power_map.transpose(), extent=extent, origin="lower", **kwargs)
        ax.set_aspect("equal")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("Detected Power [W]")
        ax.set_xlim(-0.03,0.03)
        ax.set_ylim(0, 0.231)
        ax.set_xlabel("z [m]")
        ax.set_ylabel("d [m]")


class DiscSolidAngleGainPattern:

    def __init__(self, R, d_max=10, n_r=100, n_d=1000):
        self.generate_gain_map(d_max, n_r, n_d)
        self.generate_spline()
        self.R = R

    def generate_gain_map(self, d_max, n_r, n_d):
        self.d_pos = np.linspace(0.0, d_max, n_d)
        self.r_pos = np.linspace(0.0, 1.0, n_r, endpoint=False)
        self.gains = get_disc_solid_angle_general(np.expand_dims(self.d_pos,1), self.r_pos, 1.0)/(4*np.pi)
        print(self.gains.shape)

    def generate_spline(self):
        self.spline = RectBivariateSpline(self.d_pos, self.r_pos, self.gains)

    def __call__(self, d, r, grid=False):
        return self.spline(np.abs(d)/self.R, r/self.R, grid=grid)

    def plot(self, **kwargs):
        fig, ax = plt.subplots()

        extent = [np.min(self.d_pos), np.max(self.d_pos), np.min(self.r_pos), np.max(self.r_pos)]

        power_map = self.spline(self.d_pos, self.r_pos)

        im = ax.imshow(power_map.transpose(), extent=extent, origin="lower", **kwargs)
        ax.set_aspect("equal")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("relative gain")
        #ax.set_xlim(-0.03,0.03)
        ax.set_xlim(0, 2.5)
        ax.set_xlabel("d/R")
        ax.set_ylabel("r/R")


class AntennaArray:

    def __init__(self, positions, gain_f):

        #attention !!! orientation of antenna is NOT included

        self.positions = positions
        self.gain_f = gain_f

    def get_distance(self, pos):
        return pos - np.expand_dims(self.positions, 1)

    def get_amplitude(self, dist):
        r = np.sqrt(dist[:,:,0]**2 + dist[:,:,1]**2)
        z = dist[:,:,2]
        return self.gain_f(z, r)

    @classmethod
    def make_multi_ring_array(cls, R, n_antenna, n_rings, z_min, z_max, gain_f):

        z = np.linspace(z_min, z_max, n_rings)

        angles = np.linspace(0, 1, n_antenna, endpoint=False)*2*np.pi
        x = np.cos(angles)*R
        y = np.sin(angles)*R

        xx, zz = np.meshgrid(x, z)
        yy, _ = np.meshgrid(y, z)

        positions = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))

        instance = cls(positions, gain_f)

        return instance

