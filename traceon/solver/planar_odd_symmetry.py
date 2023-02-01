import numba as nb
import numpy as np

from ..util import *

@traceon_jit
def _first_deriv_r(r_0, z_0, r, z):
    return (2*r*(z_0**2-2*z*z_0+z**2-r_0**2+r**2))/((z_0**2-2*z*z_0+z**2+r_0**2-2*r*r_0+r**2)*(z_0**2-2*z*z_0+z**2+r_0**2+2*r*r_0+r**2))

@traceon_jit
def _zeroth_deriv_z(r_0, z_0, r, z):
    return np.log(((z-z_0)**2+(r_0+r)**2)/((z-z_0)**2+(r-r_0)**2))/2

@traceon_jit
def _first_deriv_z(r_0, z_0, r, z):
    return -(4*r*r_0*(z_0-z))/((z_0**2-2*z*z_0+z**2+r_0**2-2*r*r_0+r**2)*(z_0**2-2*z*z_0+z**2+r_0**2+2*r*r_0+r**2))

@traceon_jit
def _deriv_z(v0, v1, v2, N):
    if N == -1:
        return line_integral(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], _first_deriv_r)
    if N == 0:
        return line_integral(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], _zeroth_deriv_z)
    if N == 1:
        return line_integral(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], _first_deriv_z)

