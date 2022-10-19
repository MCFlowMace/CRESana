

"""

Author: F. Thomas
Date: September 30, 2022

"""

import numpy as np
import argparse

from cresana import Simulation, SlottedWaveguideAntenna, AntennaArray, KassSimulation

parser = argparse.ArgumentParser(description='Run FSCD setup with CRESana.')
parser.add_argument('-tf', metavar='tf_file', type=str, required=True,
                    help='path to the transfer function of the 5-slot WG')
parser.add_argument('-n', metavar='n_samples', type=int, required=True,
                    help='number of samples')
parser.add_argument('-kass', metavar='kass_file', type=str, required=True,
                    help='path to the kass output file')
parser.add_argument('-out', metavar='output_file', type=str, required=True,
                    help='path to the result output file')

args = parser.parse_args()

r = 0.1
n_channels = 60
sr = 200.0e6
f_LO = 25.85e9

tf_file = args.tf
kass_file = args.kass
out_file = args.out
n_samples = args.n

kass_sim = KassSimulation(kass_file)
array = AntennaArray.make_multi_ring_array(r, n_channels, 1, 0., 0., SlottedWaveguideAntenna(5, tf_file))
simulation = Simulation(array, sr, f_LO)
signal = simulation.get_samples(n_samples, kass_sim)

np.save(out_file, signal) 
