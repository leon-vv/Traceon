import math as m

import numba as nb
import numpy as np

from ..util import *

@traceon_jit
def _first_deriv_r(r_0, z_0, r, z):
    rz2 = (r + r_0)**2 + (z - z_0)**2
    t = 4*r*r_0 / rz2
    return -r/(2*r_0*np.sqrt(rz2)) * (nb_ellipk(t) - ((z-z_0)**2 - r_0**2 + r**2) / ((z-z_0)**2 + (r-r_0)**2) * nb_ellipe(t))

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

@traceon_jit
def _get_axial_derivatives(v0, v1, v2):
    assert v0[0] == 0.0
    return line_integral(0.0, v0[1], v1[0], v1[1], v2[0], v2[1], _get_all_axial_derivatives)
    
