[![DOI](https://zenodo.org/badge/387478329.svg)](https://doi.org/10.5281/zenodo.13935567)

# CRESana
CRESana is a python package for fast simulations of CRES in an antenna array. Its main use is to sample the noise-free voltage timeseries of the antennas that are produced by a radiating trapped electron similar to the antenna array functionality of locust_mc https://github.com/project8/locust_mc. CRESana is faster due to the use of analytic results for the cyclotron radiation instead of explicit numerical calculation of the Liénard-Wiechert potential. On the electron trapping side CRESana can solve the particle motion on its own and does not have to rely on Kassiopeia https://github.com/KATRIN-Experiment/Kassiopeia. There is however an interface to Kassiopeia produced electron trajectories. With CRESana the particle motion is simulated either in traps that have exact analytic solutions or numerically in traps composed of multiple current loops. Using the numerical solution of CRESana with current loop traps is also significantly faster than doing the same in Kassiopeia. This is achieved by calculating the magnetic field itself from the analytic solution of a single current loop and using superposition for the total field. The motion calculation requires some numeric integration but exploits symmetries and periodicity to drastically reduce the computation time. It only works for electrons in static magnetic fields under the adiabatic assumption and can only account for energy loss in a 1st order approximation.

## Installation
Installation as a python package is simple via pip. Clone the repository and run `pip install .` inside the local repo. This will install two packages: `cresana`, which is the core simulation framework that contains all the classes to build simulation setups, and `cresana_samples` which contains sample setups for CRESana, allowing users to get started quickly.

## How to get started
Unfortunately as of now CRESana is overall poorly documented. It is recommended to check out the `cresana_samples` directory for same example scripts. There are standalone scripts for some specific tasks that demonstrate a bit of the electron motion simulation as well as the voltage sampling simulation and hopefully they can give a bit of an overview for how to setup simulations with the classes of the `cresana` package. In addition to the example scripts there is `cresana_samples/models.py` which contains ready to use sample setups in the form of concrete implementations of the abstract `CRESanaModel` base class. They can be used in the following way

```py
from cresana_samples import FSSTrapModel

model = FSSTrapModel(z0, I0, B0, R, n_channels, sr, f_LO, n_samples, flattened=False)
data = model(E_kin, pitch, r, t0, t_len)
```

The idea is that these models implement some basic antenna-trap configuration that is mostly fixed except for some parameters. An instance of the model can then be called with just the electron parameters and return a numpy array with the time series. The optional parameter `flattened` is `True` by default and determines if the data of all channels is to be returned in a flat 1D array or not. In the above example it returns the data in an array of the shape `(n_channels, n_samples)`. It is advised to just import existing models from the `cresana_samples` package if they fulfill your needs or otherwise adding your own implementations of the base class into your user script. Of course you can contribute yours to the package if you think they would be useful to others.
