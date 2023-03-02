from math import sqrt, cos, sin, atan2
import os.path as path

import numpy as np
from scipy.interpolate import CubicSpline

from .util import traceon_jit, QUAD_B1, QUAD_B2, QUAD_WEIGHTS
from .solver.radial_symmetry import MIN_DISTANCE_AXIS

dir_ = path.dirname(__file__)
data = path.join(dir_, 'data')
thetas_file = path.join(data, 'radial-series-3D-thetas.npy')
coefficients_file = path.join(data, 'radial-series-3D-theta-dependent-coefficients.npy')

thetas = np.load(thetas_file)
theta0 = thetas[0]
dtheta = thetas[1]-thetas[0]

radial_coefficients = np.load(coefficients_file)
thetas_interpolation_coefficients = np.load(coefficients_file)

# Maximal directional derivative being considered in 3D.
# Directly related to the shapes in 'theta_interpolation'.
DERIV_3D_MAX = 9

_shape = thetas_interpolation_coefficients.shape
assert _shape[1] == 4 # Cubic spline interpolation
assert _shape[2] == DERIV_3D_MAX//2 and _shape[3] == DERIV_3D_MAX

@traceon_jit
def radial_series_coefficients_3d(vertices, charges, zs, thetas, thetas_interpolation_coefficients):
    
    coeffs = np.zeros( (zs.size, 2, DERIV_3D_MAX//2, DERIV_3D_MAX) )
    
    for (v1, v2, v3), c in zip(vertices, charges):
    
        v1x, v1y, v1z = v1
        v2x, v2y, v2z = v2
        v3x, v3y, v3z = v3
         
        area = 1/2*sqrt(((v2y-v1y)*(v3z-v1z)-(v2z-v1z)*(v3y-v1y))**2+((v2z-v1z)*(v3x-v1x)-(v2x-v1x)*(v3z-v1z))**2+((v2x-v1x)*(v3y-v1y)-(v2y-v1y)*(v3x-v1x))**2)
        
        for i, z0 in enumerate(zs):
             
            for b1_, b2_, w in zip(QUAD_B1, QUAD_B2, QUAD_WEIGHTS):
                # Consider every quadrature point on the triangle effectively as a point charge.
                x = v1x + b1_*(v2x-v1x) + b2_*(v3x-v1x)
                y = v1y + b1_*(v2y-v1y) + b2_*(v3y-v1y)
                z = v1z + b1_*(v2z-v1z) + b2_*(v3z-v1z)
                 
                r = sqrt(x**2 + y**2 + (z-z0)**2) 
                theta = atan2((z-z0), sqrt(x**2 + y**2))
                mu = atan2(y, x)
                 
                # TODO: will this work for points exactly on the z-axis?
                index = int( (theta-theta0)/dtheta )
                #assert 0 <= index < len(thetas)
                
                t = theta-thetas[index]
                C = thetas_interpolation_coefficients[index]
                
                for nu in range(DERIV_3D_MAX//2):
                    for m in range(DERIV_3D_MAX):
                        base = t**3*C[0, nu, m] + t**2*C[1, nu, m] + t*C[2, nu, m] + C[3, nu, m]
                        r_dependence = r**(-2*nu - m - 1)
                        
                        coeffs[i, 0, nu, m] += c*area*w*base*cos(m*mu)*r_dependence
                        coeffs[i, 1, nu, m] += c*area*w*base*sin(m*mu)*r_dependence
     
    return coeffs



@traceon_jit
def compute_interpolated_potential(point, zs, coeffs):
    xp, yp, zp = point[0], point[1], point[2]
     
    if not (zs[0] <= zp < zs[-1]):
        return 0.0
     
    dz = zs[1]-zs[0]
    index = int((zp-zs[0])/dz)
     
    z_ = zp-zs[index]
    A, B = z_**3*coeffs[0,index] + z_**2*coeffs[1,index] + z_*coeffs[2,index] + coeffs[3,index]
    
    r = sqrt(xp**2 + yp**2)
    phi = atan2(yp, xp)
     
    sum_ = 0.0
     
    for nu in range(DERIV_3D_MAX//2):
        for m in range(DERIV_3D_MAX):
            sum_ += (A[nu, m]*cos(m*phi) + B[nu, m]*sin(m*phi)) * r**(m+2*nu)
        
    return sum_


@traceon_jit
def compute_interpolated_field(point, zs, coeffs):
    xp, yp, zp = point[0], point[1], point[2]
    
    if not (zs[0] <= zp < zs[-1]):
        return np.array([0.0, 0.0, 0.0])
    
    dz = zs[1]-zs[0]
    index = int((zp-zs[0])/dz)
     
    z_ = zp-zs[index]
    A, B = z_**3*coeffs[0,index] + z_**2*coeffs[1,index] + z_*coeffs[2,index] + coeffs[3,index]
    Adiff, Bdiff = 3*z_**2*coeffs[0, index] + 2*z_*coeffs[1, index] + coeffs[2, index]
     
    Ex = 0.0
    Ey = 0.0
    Ez = 0.0
    
    r = sqrt(xp**2 + yp**2)
    phi = atan2(yp, xp)
    
    if r < MIN_DISTANCE_AXIS:
        return np.array([-A[0, 1], -B[0, 1], -Adiff[0, 0]])
     
    for nu in range(DERIV_3D_MAX//2):
        for m in range(DERIV_3D_MAX):
            exp = 2*nu + m
             
            diff_r = (A[nu, m]*cos(m*phi) + B[nu, m]*sin(m*phi)) * exp*r**(exp-1)
            diff_theta = m *(-A[nu, m]*sin(m*phi) + B[nu, m]*cos(m*phi)) * r**exp
             
            Ex -= diff_r * xp/r + diff_theta * -yp/r**2
            Ey -= diff_r * yp/r + diff_theta *  xp/r**2
            Ez -= (Adiff[nu, m]*cos(m*phi) + Bdiff[nu, m]*sin(m*phi)) * r**exp
     
    return np.array([Ex, Ey, Ez])
        
    




 






