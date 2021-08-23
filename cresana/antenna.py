

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

from abc import ABC, abstractmethod
from scipy.interpolate import RectBivariateSpline
import numpy as np
import matplotlib.pyplot as plt

def fake_AM(x, sigma, A_0):
    return A_0 * np.exp(-0.5*(x/sigma)**2)

class AntennaGainPattern():

    """
    Author: R. Reimann
    """

    def __init__(self, data_paths):
        self.load_power_map(data_paths)
        self.clean_spikes()
        self.generate_spline()

    def load_power_map(self, data_pathes):
        power = None
        radius = None
        z_pos = None
        for path in data_pathes:
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
        self.z_pos = z_pos
        self.r_pos = radius
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
        self.spline = RectBivariateSpline(self.r_pos, self.z_pos, np.transpose(self.power))

    def __call__(self, r, z, grid=True):
        return self.spline(r, z, grid=grid)

    def plot(self, levels=None, levels_relative=True, **kwargs):
        fig, ax = plt.subplots(figsize=(3*2,6*2))

        extent = [np.min(self.z_pos), np.max(self.z_pos), np.min(self.r_pos), np.max(self.r_pos)]

        power_map = self.spline(self.r_pos, self.z_pos)
        print(power_map.shape)
        im = ax.imshow(power_map, extent=extent, origin="lower", **kwargs)
        ax.set_aspect("equal")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("Detected Power [W]")
        ax.set_xlim(-0.03,0.03)
        ax.set_ylim(0, 0.231)
        ax.set_xlabel("z [m]")
        ax.set_ylabel("x [m]")
        ax.grid()
        if levels is not None:
            if not levels_relative:
                levels = np.atleast_1d(levels)
                cs = ax.contour(power_map, extent=extent, levels=np.max(power_map)/10**(levels/10), colors="k")
                plt.clabel(cs, fmt={np.max(power_map)/10**(l/10): "%d dB"%l for l in levels})
            else:
                print(10**(levels/10))
                cont = [np.arange(len(row))[row > (np.max(row)/10**(levels/10))][-1] for row in power_map]
                xs = np.linspace(np.min(self.z_pos/mm), np.max(self.z_pos/mm), np.shape(power_map)[1])
                ys = np.linspace(np.min(self.r_pos/mm), np.max(self.r_pos/mm), np.shape(power_map)[0])
                plt.plot([-xs[c] for c in cont], ys, color="k")
                plt.plot([xs[c] for c in cont], ys, color="k")

