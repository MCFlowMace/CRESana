
"""

Author: F. Thomas
Date: July 19, 2021

"""


from .electronsim import Electron, AnalyticSimulation, KassSimulation
from .sampling import Simulation
from .trap import HarmonicTrap, BoxTrap, ArbitraryTrap
from .bfield import MultiCoilField, Coil
from .antenna import AntennaArray, SlottedWaveguideAntenna, IsotropicAntenna, GenericAntenna
from .model import CRESanaModel

from . import _version
__version__ = _version.get_versions()['version']
