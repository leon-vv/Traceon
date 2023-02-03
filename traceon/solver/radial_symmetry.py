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
def _get_all_axial_derivatives(_, z0, r, z):

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
    

@traceon_jit
def _second_deriv_z_inner(A, B, r_0, z_0, r, z):
    #return -(r*((-3*B*z_0**4)+A*z_0**4+12*B*z*z_0**3-4*A*z*z_0**3-18*B*z**2*z_0**2+6*A*z**2*z_0**2-2*B*r_0**2*z_0**2+A*r_0**2*z_0**2-2*A*r*r_0*z_0**2-2*B*r**2*z_0**2+A*r**2*z_0**2+12*B*z**3*z_0-4*A*z**3*z_0+4*B*r_0**2*z*z_0-2*A*r_0**2*z*z_0+4*A*r*r_0*z*z_0+4*B*r**2*z*z_0-2*A*r**2*z*z_0-3*B*z**4+A*z**4-2*B*r_0**2*z**2+A*r_0**2*z**2-2*A*r*r_0*z**2-2*B*r**2*z**2+A*r**2*z**2+B*r_0**4-2*B*r**2*r_0**2+B*r**4))/((z_0**2-2*z*z_0+z**2+r_0**2-2*r*r_0+r**2)**2*(z_0**2-2*z*z_0+z**2+r_0**2+2*r*r_0+r**2)**(3/2))
    s = np.sqrt((z-z_0)**2 + (r + r_0)**2) 
    s1 = (z_0-z)/s
    s2 = (r_0+r)**2/(s**3)
    
    ellipe_term = (r*(3*s**2-4*r*r_0)*s1**2)/(s*(s**2-4*r*r_0)**2)-(r*s2)/(s**2-4*r*r_0)
    ellipk_term = -(r*s1**2)/(s*(s**2-4*r*r_0))
     
    return A*ellipe_term + B*ellipk_term

@traceon_jit
def _second_deriv_z(r_0, z_0, r, z):
    t2 = (4*r*r_0)/((z-z_0)**2 + (r+r_0)**2)
    A = nb_ellipe(t2)
    B = nb_ellipk(t2)
    
    return _second_deriv_z_inner(A, B, r_0, z_0, r, z)


@traceon_jit
def _third_deriv_z_inner(A, B, r_0, z_0, r, z):
    s = np.sqrt((z-z_0)**2 + (r + r_0)**2) 
    s1 = (z_0-z)/s
    s2 = (r_0+r)**2/(s**3)
    s3 = -(3*(r_0+r)**2*(z_0-z))/(s**5)
      
    ellipe_term = -(r*s3)/(s**2-4*r*r_0)+(3*r*(3*s**2-4*r*r_0)*s1*s2)/(s*(s**2-4*r*r_0)**2)-(r*(11*s**4-20*r*r_0*s**2+32*r**2*r_0**2)*s1**3)/(s**2*(s**2-4*r*r_0)**3)
    ellipk_term = (r*(5*s**2-4*r*r_0)*s1**3)/(s**2*(s**2-4*r*r_0)**2)-(3*r*s1*s2)/(s*(s**2-4*r*r_0))
    
    return A*ellipe_term + B*ellipk_term
    
@traceon_jit
def _third_deriv_z(r_0, z_0, r, z):
    t2 = (4*r*r_0)/((z-z_0)**2 + (r+r_0)**2)
    A = nb_ellipe(t2)
    B = nb_ellipk(t2)
     
    return _third_deriv_z_inner(A, B, r_0, z_0, r, z)

@traceon_jit
def _fourth_deriv_z_inner(A, B, r_0, z_0, r, z):
    s = np.sqrt((z-z_0)**2 + (r + r_0)**2) 
    s1 = (z_0-z)/s
    s2 = (r_0+r)**2/(s**3)
    s3 = -(3*(r_0+r)**2*(z_0-z))/(s**5)
    s4 = (3*(r_0+r)**2*(2*z_0-2*z-r_0-r)*(2*z_0-2*z+r_0+r))/(s**7)
     
    ellipe_term = -(r*s4)/(s**2-4*r*r_0)+(4*r*(3*s**2-4*r*r_0)*s1*s3)/(s*(s**2-4*r*r_0)**2)+(3*r*(3*s**2-4*r*r_0)*s2**2)/(s*(s**2-4*r*r_0)**2)-(6*r*(11*s**4-20*r*r_0*s**2+32*r**2*r_0**2)*s1**2*s2)/(s**2*(s**2-4*r*r_0)**3)+(2*r*(25*s**6-36*r*r_0*s**4+176*r**2*r_0**2*s**2-192*r**3*r_0**3)*s1**4)/(s**3*(s**2-4*r*r_0)**4)
    ellipk_term = -(4*r*s1*s3)/(s*(s**2-4*r*r_0))-(3*r*s2**2)/(s*(s**2-4*r*r_0))+(6*r*(5*s**2-4*r*r_0)*s1**2*s2)/(s**2*(s**2-4*r*r_0)**2)-(2*r*(13*s**4-10*r*r_0*s**2+24*r**2*r_0**2)*s1**4)/(s**3*(s**2-4*r*r_0)**3)
    
    return A*ellipe_term + B*ellipk_term

@traceon_jit
def _fourth_deriv_z(r_0, z_0, r, z):
    t2 = (4*r*r_0)/((z-z_0)**2 + (r+r_0)**2)
    A = nb_ellipe(t2)
    B = nb_ellipk(t2)
     
    return _fourth_deriv_z_inner(A, B, r_0, z_0, r, z)
 

@traceon_jit
def _deriv_z(v0, v1, v2, N):
    if N == -1:
        return line_integral(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], _first_deriv_r)
    if N == 0:
        return line_integral(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], _zeroth_deriv_z)
    if N == 1:
        return line_integral(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], _first_deriv_z)
    if N == 2:
        return line_integral(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], _second_deriv_z)
    if N == 3:
        return line_integral(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], _third_deriv_z)
    if N == 4:
        return line_integral(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], _fourth_deriv_z)
    
