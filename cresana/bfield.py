

"""

Author: F. Thomas
Date: February 18, 2023

"""

from .physicsconstants import mu0

class Coil:
    
    def __init__(self, radius, z0, turns, current):
        self.radius = radius
        self.turns = turns
        self.current = current
        self.z0 = z0
        self.C = mu0*self.current*self.turns/np.pi

    def __str__(self):
        return f'd={self.radius*2}, z={self.z0}, turns={self.turns}, current={self.current}, length={self.length}'
        
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
                                   *((a**6 + r[~on_axis]**2*(2*rho[~on_axis]**2+z[~on_axis]**2) 
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
        
        return B, grad
        
    
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
