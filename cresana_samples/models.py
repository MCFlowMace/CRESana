

"""

Author: F. Thomas
Date: May 17, 2023

"""

import numpy as np

from cresana import ArbitraryTrap
from cresana import AntennaArray, IsotropicAntenna
from cresana.bfield import get_8_coil_flat_trap
from cresana.model import CRESanaModel


class FSSTrapModel(CRESanaModel):
    """
    Simulation configuration based on a flat trap design from R.Reimann.
    Trap design actually intended for use as a fieldshifting solenoid.
    Trap is combined with a single ring antenna array with isotropic antennas.
    Parameters R and n_channels define the array. Parameters z0, I0, B0 define the trap.
    Suggested trap parameters for 26GHz CRES in R=0.1m radius array:
    B0 = 0.96
    z0 = 0.15
    I0 = -6.
    """

    def __init__(self, z0, I0, B0, R, n_channels, sr, f_LO, name='DefaultFSSTrapModel', power_efficiency=1., flattened=True):
        self._n_channels = n_channels
        self._R = R
        self._z0 = z0
        self._I0 = I0
        self._B0 = B0
        CRESanaModel.__init__(self, sr, f_LO, name=name, power_efficiency=power_efficiency, flattened=flattened)

    def init_trap(self):
        self._coil_field = get_8_coil_flat_trap(self._z0, self._I0, self._B0)
        root_guess_max=0.5
        root_guess_steps=1000
        integration_steps=100
        field_line_step_size = 0.001
        root_rtol=1e-30
        b_interpolation_steps=100
        b_interpolation_order = 3

        self._trap =  ArbitraryTrap(self._coil_field, root_guess_max=root_guess_max, root_guess_steps=root_guess_steps,
                                    integration_steps=integration_steps, field_line_step_size = field_line_step_size,
                                    root_rtol=root_rtol, b_interpolation_steps=b_interpolation_steps, b_interpolation_order=b_interpolation_order, 
                                    debug=False)
        
    def init_array(self):
        self._array = AntennaArray.make_multi_ring_array(self._R, self._n_channels, 1, 0., 0., IsotropicAntenna())

    @property
    def trap(self):
        return self._trap

    @property    
    def array(self):
        return self._array
    

class ScaleFSSTrapModel(CRESanaModel):
    """
    Simulation configuration based on a flat trap design from R.Reimann.
    Trap design actually intended for use as a fieldshifting solenoid.
    Trap is combined with an antenna array that scales with the trap size equipped with isotropic antennas.
    Parameters R and L and down_scale_L, down_scale_R define the array. Using down_scale_L=down_scale_R=1
    results in an array with the densest possible antenna packing. Using integer values higher than 1 scales down the
    array. I.e. using down_scale_L=2 results in half as many rings of antennas as the full scale case, using
    down_scale_R=2 results in half as many antennas around the circumfernce as the full scale case. 
    Parameters z0, I0, B0 define the trap.
    Suggested trap parameters for 26GHz CRES in R=0.1m radius array:
    B0 = 0.96
    z0 = 0.15
    I0 = -6.
    """

    def __init__(self, z0, I0, B0, R, L, sr, f_LO, down_scale_L=1, down_scale_R=1, name='DefaultScaleFSSTrapModel', power_efficiency=1., flattened=True):
        self._down_scale_L = down_scale_L
        self.down_scale_R = down_scale_R
        self._R = R
        self._L = L
        self._z0 = z0
        self._I0 = I0
        self._B0 = B0
        CRESanaModel.__init__(self, sr, f_LO, name=name, power_efficiency=power_efficiency, flattened=flattened)

    def init_trap(self):
        self._coil_field = get_8_coil_flat_trap(self._z0, self._I0, self._B0)
        root_guess_max=0.5
        root_guess_steps=1000
        integration_steps=100
        field_line_step_size = 0.001
        root_rtol=1e-30
        b_interpolation_steps=100
        b_interpolation_order = 3

        self._trap =  ArbitraryTrap(self._coil_field, root_guess_max=root_guess_max, root_guess_steps=root_guess_steps,
                                    integration_steps=integration_steps, field_line_step_size = field_line_step_size,
                                    root_rtol=root_rtol, b_interpolation_steps=b_interpolation_steps, b_interpolation_order=b_interpolation_order, 
                                    debug=False)
        
    def init_array(self):
        #self._array = AntennaArray.make_multi_ring_array(self._R, self._n_channels, 1, 0., 0., IsotropicAntenna())
        f_min = self.f_LO-self.sr/2
        self._array = AntennaArray.make_generic_full_cylinder_array(self._L, self._R, f_min*2*np.pi, IsotropicAntenna(), 
                                                                    down_scale_R=self.down_scale_R, down_scale_L=self._down_scale_L)

    @property
    def trap(self):
        return self._trap

    @property    
    def array(self):
        return self._array