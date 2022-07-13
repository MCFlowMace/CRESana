

"""

Author: F. Thomas
Date: August 11, 2021

"""

__all__ = []

import numpy as np


def get_pos(x, y, z):
    return np.vstack((x,y,z)).transpose()
    

def norm_squared(x):
    return np.sum(x**2, axis=-1, keepdims=True)
    
    
def norm(x):
    return np.sqrt(norm_squared(x))
    

def normalize(x):
    x_norm = norm(x)
    return x/x_norm
    
    
def calc_aliased_frequencies(frequencies, sr):
	
	n = np.round(frequencies/sr)
	
	return np.abs(sr*n-frequencies)


#~ def cos_angle(a, b):
    #~ """
    #~ Returns cos(angle(a, b)) for unit vectors a and b, a is 2d array of vectors, b is 1d array of vectors
    #~ """
    
    #~ return np.einsum('ijk,ik->ij', a, b)


def project_on_plane(r, normal):
    """
    Projects the unit vectors r onto the plane with the unit normal n
    """
    
    gamma = np.einsum('ijk,ik->ij', r, normal) #cos_angle(r, normal) 
    
    return r - np.expand_dims(gamma, -1)*np.expand_dims(normal, 1)


def angle_with_orientation(a, b, n):
    """
    Calculates angles between a and b, where angles can be positive and negative
    depending on the orientation of a and b relative to the normal n.
    a, b and n assumed to be 3d np.arrays
    """
    
    cos = np.einsum('ijk,ijk->ij', a, b)
    sin = np.einsum('ijk,ijk->ij',  np.cross(a, b), n)
    
    return np.arctan2(sin, cos)
    
    
