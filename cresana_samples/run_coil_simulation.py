
"""
Author: F. Thomas
Date: March 16, 2023
"""

import numpy as np
import matplotlib.pyplot as plt

from cresana import ArbitraryTrap, MultiCoilField, Coil, Electron

E_kin = 18600.
pitch = 88.5

electron = Electron(E_kin, pitch)

B_background = 0.956

z0 = 7.5 #m
I0 = 5000 #A
r0 = 3.5

coils = [   Coil(r0, z0, 1, I0),
            Coil(r0, -z0, 1, I0)]

bfield = MultiCoilField(coils, B_background)

bfield.plot_field_2d(2.5, 10., 100, 200)

trap = ArbitraryTrap(bfield, root_guess_max=z0, root_guess_steps=1000, integration_steps=100,
                                field_line_step_size=0.001)

#solves the integration for the electron, the axial frequency is cached
simulation = trap.simulate(electron)

#query cached frequency
print('Axial frequency', trap.get_f(electron))
print('zmax', trap.find_zmax(electron))

tmax = 40.e-6
N = 2000
t = np.linspace(0,tmax, N)

#get pos(t), pitch(t), B(t), E(t), w(t) (Frequency)
pos, pitch, B, energy, w = simulation(t)

B_mean = np.mean(B)
print('Mean B', B_mean)

plt.plot(t, B)
plt.xlabel('t [s]')
plt.ylabel('B [T]')
plt.show()

plt.plot(t, pos[...,2])
plt.xlabel('t [s]')
plt.ylabel('z [m]')
plt.show()
