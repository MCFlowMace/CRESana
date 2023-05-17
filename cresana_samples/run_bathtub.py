

"""

Author: F. Thomas
Date: September 30, 2022

"""

import numpy as np
import argparse

from cresana import Simulation, SlottedWaveguideAntenna, AntennaArray, AnalyticSimulation, BathtubTrap, Electron

parser = argparse.ArgumentParser(description='Run FSCD setup with CRESana.')
parser.add_argument('-tf', metavar='tf_file', type=str, required=True,
                    help='path to the transfer function of the 5-slot WG')
parser.add_argument('-n', metavar='n_samples', type=int, required=True,
                    help='number of samples')
parser.add_argument('-out', metavar='output_file', type=str, required=True,
                    help='path to the result output file')
parser.add_argument('-E', metavar='energy', type=float, required=True,
                    help='Kinetic energy of the event')
parser.add_argument('-pitch', metavar='pitch', type=float, required=True,
                    help='pitch angle of the event in degrees')
parser.add_argument('-r', metavar='radius', type=float, required=True,
                    help='radial position of the event')
parser.add_argument('-phi', metavar='phi', type=float, required=True,
                    help='azimuth angle of the event')
parser.add_argument('-z0', metavar='z0', type=float, required=True,
                    help='z position of the event')
                    

args = parser.parse_args()

r = 0.1
n_channels = 60
sr = 200.0e6
f_LO = 25.85e9

B0 = .956 #T
L = 0.1 #1m
L0 = 2.5 #0.05m

tf_file = args.tf
out_file = args.out
n_samples = args.n
E_kin = args.E
pitch = args.pitch
r_e = args.r
phi = args.phi
z0 = args.z0

electron = Electron(E_kin, pitch, r=r_e, phi=phi, z0=z0)
trap = BathtubTrap(B0, L, L0)
array = AntennaArray.make_multi_ring_array(r, n_channels, 1, 0., 0., SlottedWaveguideAntenna(5, tf_file))

t_max = 1/sr*n_samples

analytic_sim = AnalyticSimulation(trap, electron, 2*n_samples, t_max)
simulation = Simulation(array, sr, f_LO)
signal = simulation.get_samples(n_samples, analytic_sim)

np.save(out_file, signal) 
