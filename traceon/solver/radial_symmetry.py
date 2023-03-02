import math as m

import numba as nb
import numpy as np

from ..util import *

# The derivatives in the r direction cause numerical problems around r_0 = 0.0.
# The derivatives should be 0.0 but divide by zero errors occur.
# For any derivative in the r direction closer than MIN_DISTANCE_AXIS to the
# optical axis we always return 0.0.
MIN_DISTANCE_AXIS = 1e-10

@traceon_jit
def _first_deriv_r(r_0, z_0, r, z):
    
    if abs(r_0) < MIN_DISTANCE_AXIS:
        return 0.0 # Prevent stepping into singularity
    
    s = np.sqrt((z-z_0)**2 + (r + r_0)**2) 
    s1 = (r_0+r)/s
    s2 = 1/s-(r_0+r)**2/s**3
    
    t = (4*r*r_0)/s**2
      
    A = nb_ellipe(t)
    B = nb_ellipk(t)
     
    ellipe_term = -(2*r*r_0*s1-r*s)/(2*r_0*s**2-8*r*r_0**2)
    ellipk_term = -r/(2*r_0*s)

    return A*ellipe_term + B*ellipk_term


@traceon_jit
def _zeroth_deriv_z(r_0, z_0, r, z):
    rz2 = (r + r_0)**2 + (z - z_0)**2
    t = 4*r*r_0 / rz2
    return nb_ellipk(t) * r / np.sqrt(rz2)

@traceon_jit
def _first_deriv_z(r_0, z_0, r, z):
    rz2 = (r + r_0)**2 + (z - z_0)**2
    t = 4*r*r_0 / rz2
    return r*(z-z_0)*nb_ellipe(t) / ( ((z-z_0)**2 + (r-r_0)**2)*np.sqrt(rz2) )

@traceon_jit
def _get_all_axial_derivatives(r0, z0, r, z):

    R = m.sqrt( (z0 - z)**2 + r**2)

    D = np.empty(9, np.float64)
    D[0] = 1/R
    D[1] = -(z0-z)/R**3
    
    for n in range(1,8):
        D[n+1] = -1/R**2 * ((2*n + 1)*(z0 - z)*D[n] + n**2 * D[n-1])

    return np.pi*r/2 * D

