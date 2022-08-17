

"""

Author: F. Thomas
Date: July 26, 2021

"""

__all__ = []

from abc import ABC, abstractmethod

from scipy.signal import sawtooth
from scipy.interpolate import interp1d
import numpy as np
import uproot

from .physicsconstants import speed_of_light, E0_electron
from .utility import get_pos
from .cyclotronphysics import get_relativistic_velocity
from .sampling import Clock

def get_x(R, phi):
    return R*np.cos(phi)


def get_y(R, phi):
    return R*np.sin(phi)


def gradB_phase(t, omega, phi):
    return t*omega + phi
    
    
def find_nearest_samples(t1, t2):
    ind = np.searchsorted((t2[1:]+t2[:-1])/2, t1)
    last = np.searchsorted(ind, t2.shape[0]-1)

    return t2[ind[:last]], ind[:last]
    
    
def find_nearest_samples2d(t1, t2):
    t = np.empty(shape=t1.shape)
    ind = np.empty(shape=t1.shape, dtype=np.int64)
    
    for i in range(t1.shape[0]):
        t[i], ind[i] = find_nearest_samples(t1[i], t2)
        
    return t, ind


class Electron:
    """Represents the initial state of an electron.

    Attributes
    ----------
    E_kin : float
        Initial kinetic energy.
    pitch : float
        Initial pitch angle at trap minimum.
    v0    : float
        Initial velocity.
    """

    def __init__(self, E_kin, pitch, r=0, phi=0, z0=0, v_phi=0):
        self._E_kin = E_kin
        self._pitch = pitch/180*np.pi
        self._x0 = r*np.cos(phi)
        self._y0 = r*np.sin(phi)
        self._z0 = z0
        self._v_phi = v_phi
        self._v0 = get_relativistic_velocity(E_kin)

    @property
    def E_kin(self):
        return self._E_kin

    @property
    def pitch(self):
        return self._pitch

    @property
    def v0(self):
        return self._v0

    def __repr__(self):
        return ("Kinetic energy : {0:10.4f} eV \n".format(self._E_kin)
                +"Pitch angle:  {0:8.4f} ° \n".format(self._pitch/np.pi*180)
                +"Velocity:  {0:14.2f} m/s \n".format(self._v0)
                +"X0:  {0:6.4f} m \n".format(self._x0)
                +"Y0:  {0:6.4f} m \n".format(self._y0)
                +"Z0:  {0:6.4f} m \n".format(self._z0)
                +"Velocity angle:  {0:6.4f} \n".format(self._v_phi)
                )


class ElectronSim:
    """Represents an electron simulation result.

    Attributes
    ----------
    coords : numpy.ndarray
        Trajectory of the electron.
    t      : numpy.ndarray
        Sampling times.
    B_vals : numpy.ndarray
        Absolute B-field experienced by the electron.
    """

    def __init__(self, coords, t, B_vals, E_kin, pitch, B_direction):
        self.coords = coords
        self.t = t
        self.B_vals = B_vals
        self.E_kin = E_kin
        self.pitch = pitch
        self.B_direction = B_direction
        

class ElectronSimulator:
    
    def __init__(self):
        pass
        
    @abstractmethod
    def simulate(self, t):
        pass
        
    @abstractmethod
    def get_sample_time_trajectory(self, t_sample):
        pass
        
    def __call__(self, t):
        
        coords, t_traj, B, E_kin, pitch, B_direction = self.simulate(t)
        self.enforce_causality(t, t_traj, B, E_kin, pitch)
        
        return ElectronSim(coords, t_traj, B, E_kin, 
                                    pitch, self.electron_sim.B_direction)
        
    def enforce_causality(self, t, t_traj, B, E, pitch):
        
        # setting to zero adds a small error for the initial phase in the phase integral
        # since this way the integral starts from t_ret=0 (which is correct) but 
        # it is assumed that omega(t_ret=0) = 0
        # the alternative solution of passing only the causal indices to the integral
        # has a wrong initial phase as well since it starts the integral from
        # some t_ret>0. The correct solution would be to find the correct value of omega(t_ret=0)
        # and integrating from there
        ind_non_causal = t<0
        B[ind_non_causal] = 0
        t_traj[ind_non_causal] = 0
        E[ind_non_causal] = 1.0
        pitch[ind_non_causal] = np.pi/2
        
        
class KassSimulation(ElectronSimulator):
    
    def __init__(self, file_name, interpolation='spline', decimation_factor=1):
        ElectronSimulator.__init__(self)
        
        if interpolation != 'nearest' and interpolation != 'spline':
            raise ValueError('interpolation must be either "nearest" or "spline"')
            
        self.decimation_factor = decimation_factor
        self.interpolation = interpolation
        self._read_kass_sim(file_name)
        self._interpolate()
        
    def simulate(self, t):
        
        print('Using Kassiopeia simulated trajectory')
        
        if self.interpolation=='spline':
            
            print('Spline interpolation')
            
            t_traj = t
            B = self.B_f(t)
            pitch = self.pitch_f(t)
            E_kin = self.E_f(t)
            coords = self.coords_f(t)
            
        else:
            
            print('Nearest neighbor interpolation')

            t_traj, sample_ind = find_nearest_samples2d(t, self.electron_sim.t)
            
            B = self.electron_sim.B_vals[sample_ind]
            pitch = self.electron_sim.pitch[sample_ind]
            E_kin = self.electron_sim.E_kin[sample_ind]
            coords = self.electron_sim.coords[sample_ind]

        return coords, t_traj, B, E_kin, pitch, self.electron_sim.B_direction
        
    def get_sample_time_trajectory(self, t_sample):
        
        if self.interpolation=='spline':
            t = t_sample
            coords = self.coords_f(t)
        else:
            #nearest neighbor interpolation
            t, sample_ind = find_nearest_samples(t_sample, self.electron_sim.t)
            coords = self.electron_sim.coords[sample_ind]
        
        return t, coords
        
    def _interpolate(self):
        
        if self.interpolation == 'spline':
            self.coords_f = interp1d(self.electron_sim.t, self.electron_sim.coords, 
                        kind='cubic', axis=0, bounds_error=False, fill_value='extrapolate')
                        
            self.B_f = interp1d(self.electron_sim.t, self.electron_sim.B_vals, kind='cubic', 
                            bounds_error=False, fill_value='extrapolate')
            self.pitch_f = interp1d(self.electron_sim.t, self.electron_sim.pitch, kind='cubic', 
                            bounds_error=False, fill_value='extrapolate')
            self.E_f = interp1d(self.electron_sim.t, self.electron_sim.E_kin, kind='cubic', 
                            bounds_error=False, fill_value='extrapolate')
        
    def _read_kass_sim(self, name):
        file_input = uproot.open(name)

        tree = file_input['component_step_world_DATA']
        branches = tree.arrays()

        def data(key):
            return np.array(branches[key][:-1:self.decimation_factor])

        t = data('time')

        x = data('guiding_center_position_x')
        y = data('guiding_center_position_y')
        z = data('guiding_center_position_z')

        B_x = data('magnetic_field_x')
        B_y = data('magnetic_field_y')
        B_z = data('magnetic_field_z')

        px = data('momentum_x')
        py = data('momentum_y')
        pz = data('momentum_z')

        E_kin = data('kinetic_energy')

        B_vals = np.sqrt(B_x**2 + B_y**2 + B_z**2)
        p = np.sqrt(px**2 + py**2 + pz**2)
        pitch = np.arccos(pz/p)

        coords = get_pos(x, y, z)
        
        B_direction = np.array([B_x[0], B_y[0], B_z[0]])/B_vals[0]

        self.electron_sim = ElectronSim(coords, t, B_vals, E_kin, pitch, B_direction)



class AnalyticSimulation(ElectronSimulator):
    
    def __init__(self, trap, electron, N, t_max):
        ElectronSimulator.__init__(self)
        
        self.coords_f = trap.trajectory(electron)
        self.electron = electron
        self.B_f = trap.B_field
        self.pitch_f = trap.pitch(electron)
        self._get_initial_sim(N, t_max)
        
    def simulate(self, t):
        
        coords = self.coords_f(t)
        B_vals = self.B_f(coords[...,2])
        pitch = self.pitch_f(t)
        B_direction = np.array([0,0,1])
       
        E_kin = np.ones_like(pitch)*self.electron._E_kin # energy loss?!
        
        return coords, t, B_vals, E_kin, pitch, B_direction
        
    def get_sample_time_trajectory(self, t_sample):
        
        t = t_sample
        coords = self.coords_f(t)
        
        return t, coords

    def _get_initial_sim(self, N, t_max):
        
        dt = t_max/N
        sampling_rate = 1/dt
        clock = Clock(sampling_rate)
        data = self.simulate(clock(N))
        self.electron_sim = ElectronSim(*data)
        
