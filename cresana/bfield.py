

"""

Author: F. Thomas
Date: February 18, 2023

"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator, make_interp_spline
from scipy.special import ellipk, ellipe, ellipkm1
from warnings import warn
import matplotlib.pyplot as plt

from .physicsconstants import mu0

class Coil:
    
    def __init__(self, radius, z0, turns, current):
        self.radius = radius
        self.turns = turns
        self.current = current
        self.z0 = z0
        self.C = mu0*self.current*self.turns/np.pi

    def __str__(self):
        return f'd={self.radius*2}, z={self.z0}, turns={self.turns}, current={self.current}'
        
    def __repr__(self):
        return str(self)
    
    def evaluate_B(self, pos, derivatives=False):
        # Analytic part based on
        # https://ntrs.nasa.gov/api/citations/20140002333/downloads/20140002333.pdf
        rho = pos[...,0]
        z = pos[...,1] - self.z0
        
        B = np.empty_like(pos)
        Brho = B[...,0]
        Bz = B[...,1]
        
        C = self.C
        a = self.radius
        r = np.sqrt(rho**2 + z**2)
        alpha = np.sqrt(self.radius**2 + r**2 - 2*self.radius*rho)
        beta  = np.sqrt(self.radius**2 + r**2 + 2*self.radius*rho)
        k2 = 1 - alpha**2/beta**2
        E_k2 = ellipe(k2)
        K_k2 = np.empty_like(k2)
        greater = np.abs(k2-1)>0.1
        K_k2[greater] = ellipk(k2[greater])
        K_k2[~greater] = ellipkm1(1-k2[~greater])
        
        on_axis = rho==0
        Brho[on_axis] = 0
    
        Brho[~on_axis] = (C*rho[~on_axis]*z[~on_axis]/(2*alpha[~on_axis]**2*beta[~on_axis]*rho[~on_axis]**2)
                        *((a**2 + r[~on_axis]**2)*E_k2[~on_axis]-alpha[~on_axis]**2*K_k2[~on_axis]))

        Bz[...] = C/(2*alpha**2*beta)*((a**2 - r**2)*E_k2 + alpha**2*K_k2)

        if derivatives:
            dB = np.empty(pos.shape[:-1]+(3,))
            dBrho_rho = dB[...,0]
            dBrho_z = dB[...,1]
            dBz_z = dB[...,2]
            
            dBrho_rho[on_axis] = 0
            dBrho_z[on_axis] = 0
            
            dBrho_rho[~on_axis] = (-C*z[~on_axis]/(2*rho[~on_axis]**2*alpha[~on_axis]**4*beta[~on_axis]**3)
                                   *((a**6 + r[~on_axis]**4*(2*rho[~on_axis]**2+z[~on_axis]**2) 
                                      + a**4*(3*z[~on_axis]**2 - 8*rho[~on_axis]**2) 
                                      + a**2*(5*rho[~on_axis]**4 - 4*rho[~on_axis]**2*z[~on_axis]**2 
                                              + 3*z[~on_axis]**4))*E_k2[~on_axis]
                                     - alpha[~on_axis]**2*(a**4 - 3*a**2*rho[~on_axis]**2 + 2*rho[~on_axis]**4 
                                                           + (2*a**2 + 3*rho[~on_axis]**2)*z[~on_axis]**2 
                                                           + z[~on_axis]**4)*K_k2[~on_axis]))

            dBrho_z[~on_axis] = (C/(2*rho[~on_axis]*alpha[~on_axis]**4*beta[~on_axis]**3)
                                   *(((a**2 + rho[~on_axis]**2)*(z[~on_axis]**4 + (a**2-rho[~on_axis]**2)**2)
                                      +2*z[~on_axis]**2*(a**4 - 6*a**2*rho[~on_axis]**2 
                                                         + rho[~on_axis]**4))*E_k2[~on_axis]
                                     -alpha[~on_axis]**2*((a**2 - rho[~on_axis]**2)**2 
                                                          + (a**2 + rho[~on_axis]**2)*z[~on_axis]**2)*K_k2[~on_axis]))

            dBz_z[...] = C*z/(2*alpha**4*beta**3)*((6*a**2*(rho**2 - z**2) - 7*a**4 + (rho**2 + z**2)**2)*E_k2 
                                              + alpha**2*(a**2 - rho**2 - z**2)*K_k2)
            
            return B, dB
        
        return B
        
        
class MultiCoilField:
    
    def __init__(self, coils, background_field):
        self.coils = coils
        self.background_field = background_field
        self.Bmag_f = None
        self.B_f = None
        
        
    def _is_potential_well(self):
        n = 0
        for c in self.coils:
            if c.current<0:
                n+=1
        
        if n!=0 and n!=len(self.coils):
            warn('It seems your trap is using coils with positive and negative currents!'
            + 'This could be what you want, but "MultiCoilField.get_B_max was implemented'
            + ' under the assumption that this does not happen so please check this again!')
            
        # the trap is a potential well if all coils have negative current
        return n==len(self.coils)
        
    def get_B_max(self, r):
        
        if self._is_potential_well():
            return self.background_field
        else:
            B_max = 0
            for c in self.coils:
                b = self.evaluate_B(np.array([r, c.z0]))
                b_mag = np.sqrt(b[...,0]**2 + b[...,1]**2)
                if b_mag>B_max:
                    B_max = b_mag
            return B_max
        
    def evaluate_B(self, pos, derivatives=False):
        
        if not derivatives:
            
            B = np.zeros_like(pos)

            for c in self.coils:
                B += c.evaluate_B(pos)

            B[...,1] += self.background_field
            
            return B
        B = np.zeros_like(pos)
        dB = np.zeros(pos.shape[:-1]+(3,))

        for c in self.coils:
            B_, dB_ = c.evaluate_B(pos, derivatives=True)
            B += B_
            dB += dB_

        B[...,1] += self.background_field

        return B, dB

    def get_grad_mag(self, pos):
        
        B, dB = self.evaluate_B(pos, derivatives=True)
        B_mag = np.sqrt(B[...,0]**2 + B[...,1]**2)
        Br = B[...,0]
        Bz = B[...,1]
        
        dBrho_rho = dB[...,0]
        dBrho_z = dB[...,1]
        dBz_z = dB[...,2]
        dBz_rho = dBrho_z
        
        grad = np.empty(pos.shape[:-1])
        B0 = B_mag==0
        grad[B0] = 0
        grad[~B0] = 1/B_mag[~B0]**2*(Bz[~B0]**2*dBz_rho[~B0] - Br[~B0]**2*dBrho_z[~B0]
                                    +Bz[~B0]*Br[~B0]*(dBrho_rho[~B0]-dBz_z[~B0]))
                                    
        curv = np.empty(pos.shape[:-1])
        
        curv[B0] = 0
        curv[~B0] = 1/B_mag[~B0]**3*(Bz[~B0]*(Br[~B0]*dBrho_rho[~B0] + Bz[~B0]*dBrho_z[~B0]) - Br[~B0]*(Bz[~B0]*dBz_rho[~B0] + Bz[~B0]*dBz_rho[~B0]))
        
        return B_mag, grad, curv
    
    def gen_field_line(self, r0, z0, dt, zmax):
        
        pos_z = [z0]
        pos_r = [r0]
        while pos_z[-1]<zmax:
            B = self.evaluate_B(np.array([pos_r[-1], pos_z[-1]]))
            B_mag = np.sqrt(B[0]**2 + B[1]**2)
            dz = B[1]/B_mag*dt
            dr = B[0]/B_mag*dt
            pos_z.append(pos_z[-1]+dz)
            pos_r.append(pos_r[-1]+dr)
            
        line = make_interp_spline(pos_z, pos_r, bc_type='clamped')

        return line
    
    def interpolate(self, rmax, zmax, nr, nz):
        
        dr = rmax/nr
        dz = 2*zmax/nz
        
        r = np.arange(0, rmax+dr, dr)
        z = np.arange(-zmax, zmax+dz, dz)
        grid = np.moveaxis(np.mgrid[(slice(0,rmax+dr,dr),slice(-zmax,zmax+dz,dz))],0,-1)
        
        B = self.evaluate_B(grid)
        B_mag = np.sqrt(B[...,0]**2 + B[...,1]**2)
        
        self.B_f = RegularGridInterpolator((r,z), B, method='cubic')
        self.Bmag_f = RegularGridInterpolator((r,z), B_mag, method='cubic')
    
    def to_cylindric(self, pos):
        pos_c = np.empty(pos.shape[:-1]+(2,))

        rho = pos_c[...,0]
        z = pos_c[...,1]

        rho[...] = np.sqrt(pos[...,0]**2+pos[...,1]**2)

        z[...] = pos[...,2]
        return pos_c
    
    def get_B(self, pos):
        if self.Bmag_f is None:
            raise RuntimeError('Interpolation function does not exist. Call MultiCoilField.interpolate!')
            
        pos_c = self.to_cylindric(pos)
        
        print('pos_c shape', pos_c.shape)
        
        return self.Bmag_f((pos_c[...,0], pos_c[...,1]))
            
    def get_Br(self, pos):
        if self.B_f is None:
            raise RuntimeError('Interpolation function does not exist. Call MultiCoilField.interpolate!')
            
        pos_c = self.to_cylindric(pos)
        
        return self.B_f((pos_c[...,0], pos_c[...,1]))[...,0]
            
    def get_Bz(self, pos):
        if self.B_f is None:
            raise RuntimeError('Interpolation function does not exist. Call MultiCoilField.interpolate!')
            
        pos_c = self.to_cylindric(pos)
        
        return self.B_f((pos_c[...,0], pos_c[...,1]))[...,1]
        
    def visualize(self, name=None):
    
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.view_init(vertical_axis='y')
        ax.set_title(f'Background field {self.background_field:5.3f} T')

        t = np.linspace(0,2*np.pi, 1000)
        
        for c in self.coils:
            
            r_coil = c.radius
            z0 = c.z0
            
            xline = r_coil*np.cos(t)
            yline = r_coil*np.sin(t)
            zline = np.ones_like(t)*z0
            ax.plot3D(xline, yline, zline)
            
        if name is not None:
            plt.savefig(name+'.png', dpi=600)
            
        plt.show()
        
    def plot_field_2d(self, rmax, zmax, nr, nz, name=None):
        
        dr = rmax/nr
        dz = 2*zmax/nz
        grid = np.moveaxis(np.mgrid[(slice(0,rmax+dr,dr),slice(-zmax,zmax+dz,dz))],0,-1)
        
        aspect = 2*zmax/rmax
        
        B_map, grad_map, curv_map = self.get_grad_mag(grid)
        
        im = plt.imshow(B_map, origin='lower', extent=(grid[0,0][1], grid[0,-1][1], grid[0,0][0],grid[-1,0][0]),zorder=2)
        cbar = plt.colorbar(im)                                                           
        cbar.set_label('B [T]')
        plt.xlabel('z[m]')
        plt.ylabel('r[m]')
        plt.gca().set_aspect(aspect)
        plt.tight_layout()
        
        if name is not None:
            plt.savefig('B_map_'+name+'.png', dpi=600)
        plt.show()

        im = plt.imshow(grad_map, origin='lower', extent=(grid[0,0][1], grid[0,-1][1], grid[0,0][0],grid[-1,0][0]),zorder=2)
        cbar = plt.colorbar(im)                                                           
        cbar.set_label(r'$\nabla B$ [T/m]')
        plt.xlabel('z[m]')
        plt.ylabel('r[m]')
        plt.gca().set_aspect(aspect)
        plt.tight_layout()
                
        if name is not None:
            plt.savefig('B_grad_map_'+name+'.png', dpi=600)
        plt.show()
        
        im = plt.imshow(curv_map, origin='lower', extent=(grid[0,0][1], grid[0,-1][1], grid[0,0][0],grid[-1,0][0]),zorder=2)
        cbar = plt.colorbar(im)                                                           
        cbar.set_label(r'$B \times (B \cdot \nabla) B/B^3$ [1/m]')
        plt.xlabel('z[m]')
        plt.ylabel('r[m]')
        plt.gca().set_aspect(aspect)
        plt.tight_layout()
        
        if name is not None:
            plt.savefig('B_curv_map_'+name+'.png', dpi=600)
        plt.show()
        
    def plot_field_profile(self, r0, z0, nz=100, nr=1, x_ax='z', name=None):
        
        z = np.linspace(-z0, z0, nz) if nz>1 else [z0]
        r = np.linspace(0, r0, nr) if nr>1 else [r0]
        
        if x_ax=='z':
            x = z
            loop_vals = r
            
            def pos(r0):
                r_vals = np.ones_like(z)*r0
                return np.stack((r_vals, z), axis=-1)
            
            xlabel= 'z[m]'
            legend_label='r'
        else:
            x = r
            loop_vals = z[z>=0]
            
            def pos(z0):
                z_vals = np.ones_like(r)*z0
                return np.stack((r, z_vals), axis=-1)
                
            xlabel='r[m]'
            legend_label='z'
            
        fig_B, ax_B = plt.subplots()
        fig_grad, ax_grad = plt.subplots()
        fig_curv, ax_curv = plt.subplots()
        
        ax_B.set_xlabel(xlabel)
        ax_B.set_ylabel('B [T]')
        
        ax_grad.set_xlabel(xlabel)
        ax_grad.set_ylabel(r'$\nabla B$ [T/m]')
        
        ax_curv.set_xlabel(xlabel)
        ax_curv.set_ylabel(r'$B \times (B \cdot \nabla) B/B^3$ [1/m]')
            
        for val in loop_vals:
            
            B, grad, curv = self.get_grad_mag(pos(val))

            ax_B.plot(x, B, label=f'{legend_label}={val:6.2f}m')
            ax_grad.plot(x, grad, label=f'{legend_label}={val:6.2f}m')
            ax_curv.plot(x, curv, label=f'{legend_label}={val:6.2f}m')
        
        
        ax_B.legend()
        ax_grad.legend()
        ax_curv.legend()

        fig_B.tight_layout()
        fig_grad.tight_layout()
        fig_curv.tight_layout()
        
        if name is not None:
            fig_B.savefig(name+'_mag_'+legend_label+'.png', dpi=600)
            fig_grad.savefig(name+'_grad_'+legend_label+'.png', dpi=600)
            fig_curv.savefig(name+'_curv_'+legend_label+'.png', dpi=600)
            
        plt.show()

    def plot_field_lines(self,r0, dr, nr, zmax, dz=0.1, z0=0., nz=1000, name=None):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        r_vals = np.linspace(r0-dr, r0+dr, nr)

        z = np.linspace(-zmax, zmax, nz)
        
        for r in r_vals:
            field_line = self.gen_field_line(r, z0, dz, zmax)

            
            r_ = field_line(np.abs(z))

            plt.plot(z,r_, c=color)
            
        plt.xlabel('z [m]')
        plt.ylabel('r [m]')
        
        if name is not None:
            plt.savefig(name+'_field_lines.png', dpi=600)
            
        plt.show()
                

def get_8_coil_flat_trap(z0, I0, B_background):
    """Get a MultiCoildField instance with 8 coils that produces a 
    trap with a flat central field region.
    
    ----------
    z0 : float
        Approximate position of the potential wall in m.
    I0 : float
        Factor that scales the current used for the coils. All currents
        for all coils scale with this parameter.
    B_background    : float
        Background magnetic field in T.
    """

    # 8 coils on a sphere # Williams-Cain
    b1 = z0
    I1 = I0*9/4/0.9999914835119487*3*5 

    b2 = b1
    b3 = b1
    b4 = b1
    x1 = 0.1652754
    x2 = 0.4779250
    x3 = 0.7387739
    x4 = 0.9195342

    I2 = 0.891626*I1
    I3 = 0.686604*I1
    I4 = 0.406992*I1
    Z1 = b1*x1
    R1 = b1*np.sqrt(1-x1**2)
    Z2 = b2*x2
    R2 = b2*np.sqrt(1-x2**2)
    Z3 = b3*x3
    R3 = b3*np.sqrt(1-x3**2)
    Z4 = b4*x4
    R4 = b4*np.sqrt(1-x4**2)

    coils = [   Coil(R1, Z1, 1, I1),
                Coil(R1, -Z1, 1, I1),
                Coil(R2, Z2, 1, I2),
                Coil(R2, -Z2, 1, I2),
                Coil(R3, Z3, 1, I3),
                Coil(R3, -Z3, 1, I3),
                Coil(R4, Z4, 1, I4),
                Coil(R4, -Z4, 1, I4)]

    return MultiCoilField(coils, B_background)
    
    
