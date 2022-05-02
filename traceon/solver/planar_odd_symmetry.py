import numba as nb
import numpy as np

from .util import *

@traceon_jit
def _first_deriv_r(r_0, z_0, r, z):
    #return (r_0+r)/((z-z_0)**2+(r_0+r)**2)**(3/2)+(r-r_0)/((z-z_0)**2+(r-r_0)**2)**(3/2)
    return (2*r*(z_0**2-2*z*z_0+z**2-r_0**2+r**2))/((z_0**2-2*z*z_0+z**2+r_0**2-2*r*r_0+r**2)*(z_0**2-2*z*z_0+z**2+r_0**2+2*r*r_0+r**2))

@traceon_jit
def _zeroth_deriv_z(r_0, z_0, r, z):
    #return 1/np.sqrt((z-z_0)**2+(r-r_0)**2)-1/np.sqrt((z-z_0)**2+(r_0+r)**2)
    return np.log(((z-z_0)**2+(r_0+r)**2)/((z-z_0)**2+(r-r_0)**2))/2

@traceon_jit
def _first_deriv_z(r_0, z_0, r, z):
    return -(4*r*r_0*(z_0-z))/((z_0**2-2*z*z_0+z**2+r_0**2-2*r*r_0+r**2)*(z_0**2-2*z*z_0+z**2+r_0**2+2*r*r_0+r**2))
    #return (z-z_0)/((z-z_0)**2+(r-r_0)**2)**(3/2)-(z-z_0)/((z-z_0)**2+(r_0+r)**2)**(3/2)

@traceon_jit
def _deriv_z_far_away(v0, v1, v2, N):
    mid = (v1+v2)/2
    r, z = mid[0], mid[1]
    r0, z0 = v0[0], v0[1]
    
    length = norm(v1[0]-v2[0], v1[1]-v2[1])
    
    if N == -1:
        return _first_deriv_r(r0, z0, r, z)*length
    elif N == 0:
        return _zeroth_deriv_z(r0, z0, r, z)*length
    elif N == 1:
        return _first_deriv_z(r0, z0, r, z)*length

@traceon_jit
def _deriv_z_close(v0, v1, v2, Nd):
    N = N_FACTOR*WIDTHS_FAR_AWAY
    r0, z0 = v0[0], v0[1]
    r = np.linspace(v1[0], v2[0], N)
    z = np.linspace(v1[1], v2[1], N)
     
    ds = norm(r[1]-r[0], z[1]-z[0])
     
    if Nd == -1:
        points = _first_deriv_r(r0, z0, r, z)
    if Nd == 0:
        points = _zeroth_deriv_z(r0, z0, r, z)
    if Nd == 1:
        points = _first_deriv_z(r0, z0, r, z)
     
    return simps(points, ds)
 
@traceon_jit
def _deriv_z(v0, v1, v2, N):
    mid = (v1+v2)/2
    width = norm(v1[0]-v2[0], v1[1]-v2[1])
    distance = norm(mid[0]-v0[0], mid[1]-v0[1])
    
    if distance > WIDTHS_FAR_AWAY*width:
        return _deriv_z_far_away(v0, v1, v2, N)
    else:
        return _deriv_z_close(v0, v1, v2, N)


