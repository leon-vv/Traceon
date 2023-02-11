from math import sqrt

import numba as nb
import numpy as np

from ..util import *

@traceon_jit
def _first_deriv_x(x0, y0, z0, x, y, z):
    r = sqrt( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 )
    return (x-x0)/(4*r**3)

@traceon_jit
def _first_deriv_y(x0, y0, z0, x, y, z):
    r = sqrt( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 )
    return (y-y0)/(4*r**3)

@traceon_jit
def _first_deriv_z(x0, y0, z0, x, y, z):
    r = sqrt( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 )
    return (z-z0)/(4*r**3)

@traceon_jit
def _zeroth_deriv(x0, y0, z0, x, y, z):
    r = sqrt( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 )
    return 1/(4*r)
