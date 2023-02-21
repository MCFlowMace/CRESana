

"""

Author: F. Thomas
Date: July 19, 2021

"""

__all__ = []

import numpy as np


### https://pdg.lbl.gov/2018/reviews/rpp2018-rev-phys-constants.pdf
# https://pdg.lbl.gov/2021/tables/rpp2021-sum-leptons.pdf
#https://pdg.lbl.gov/2021/reviews/rpp2020-rev-phys-constants.pdf

#pdg 2021
E0_electron = 0.51099895e6
speed_of_light = 299792458 # speed of light m/s
epsilon0 = 55.26349406e6 # electric field constant e^2/(eV m)
ev = 1.602176634e-19
mu0 = 1.00000000055*4*np.pi*1e-7 #magnetic field constant N/A^2

#~ #kass-locust
#~ E0_electron = 0.510998918e6
#~ speed_of_light = 299792458 # speed of light
#~ epsilon0 = 55.26349406e6 # electric field constant e^2/(eV m) (this is pdg2021 not kassiopeia)
#~ ev = 1.60217653e-19

#~ #epsilon0 = 8.854187817e-12 #this is kassiopeia but conversion to e^2/(eV m) necessary
