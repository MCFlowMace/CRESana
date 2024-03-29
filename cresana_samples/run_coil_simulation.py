
"""
Author: F. Thomas
Date: March 16, 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import time

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
bfield.plot_field_lines(1., 0.0005, 7, z0, dz=0.01, z0=0., nz=1000)

trap = ArbitraryTrap(bfield, root_guess_max=z0, root_guess_steps=1000, integration_steps=100,
                                field_line_step_size=0.001)

start = time.time()
Bmax = bfield.get_B_max(1.)
end = time.time()
print('Bmax takes [s]', end-start)

print('Bmax', bfield.get_B_max(0.))
print('Bmin', bfield.get_grad_mag(np.array([[0., 0., 0.]])))
print('zmax', trap.find_zmax(electron))

#solves the integration for the electron, the axial frequency is cached
start = time.time()
simulation = trap.simulate(electron)
end = time.time()
print('sim takes [s]', end-start)

#query cached frequency
print('Axial frequency', trap.get_f(electron))

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
