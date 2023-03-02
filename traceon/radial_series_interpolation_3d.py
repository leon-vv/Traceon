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
radial_coefficients = np.load(coefficients_file)

thetas_to_radial_coefficients_3d = CubicSpline(thetas, radial_coefficients)

thetas_interpolation_coefficients = thetas_to_radial_coefficients_3d.c

# Maximal directional derivative being considered in 3D.
# Directly related to the shapes in 'theta_interpolation'.
DERIV_3D_MAX = 9

SERIES_FACTORS = np.array([
[1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0, 1.0e0],
[ -2.5e-1, -1.25e-1, -8.333333333333333e-2, -6.25e-2, -5.0e-2, -4.166666666666667e-2, -3.571428571428571e-2, -3.125e-2, -2.777777777777778e-2, -2.5e-2, -2.272727272727273e-2, -2.083333333333333e-2, -1.923076923076923e-2],
[ 1.5625e-2, 5.208333333333333e-3, 2.604166666666667e-3, 1.5625e-3, 1.041666666666667e-3, 7.44047619047619e-4, 5.580357142857143e-4, 4.340277777777778e-4, 3.472222222222222e-4, 2.840909090909091e-4, 2.367424242424242e-4, 2.003205128205128e-4, 1.717032967032967e-4],
[ -4.340277777777778e-4, -1.085069444444444e-4, -4.340277777777778e-5, -2.170138888888889e-5, -1.240079365079365e-5, -7.750496031746032e-6, -5.166997354497354e-6, -3.616898148148148e-6, -2.63047138047138e-6, -1.972853535353535e-6, -1.517579642579643e-6, -1.192384004884005e-6, -9.539072039072039e-7],
[ 6.781684027777778e-6, 1.356336805555556e-6, 4.521122685185185e-7, 1.937624007936508e-7, 9.68812003968254e-8, 5.382288910934744e-8, 3.229373346560847e-8, 2.055055765993266e-8, 1.370037177328844e-8, 9.484872766122766e-9, 6.774909118659119e-9, 4.968266687016687e-9, 3.726200015262515e-9],
[ -6.781684027777778e-8, -1.130280671296296e-8, -3.229373346560847e-9, -1.211015004960317e-9, -5.382288910934744e-10, -2.691144455467372e-10, -1.467896975709476e-10, -8.562732358305275e-11, -5.269373758957092e-11, -3.387454559329559e-11, -2.25830303955304e-11, -1.552583339692715e-11, -1.095941180959563e-11],
[ 4.709502797067901e-10, 6.72786113866843e-11, 1.681965284667108e-11, 5.606550948890359e-12, 2.242620379556143e-12, 1.019372899798247e-12, 5.096864498991235e-13, 2.744465499456819e-13, 1.568265999689611e-13, 9.409595998137665e-14, 5.880997498836041e-14, 3.805351322776262e-14, 2.536900881850841e-14],
[ -2.402807549524439e-12, -3.003509436905549e-13, -6.674465415345665e-14, -2.0023396246037e-14, -7.281234998558907e-15, -3.033847916066211e-15, -1.400237499722867e-15, -7.001187498614334e-16, -3.733966665927645e-16, -2.1003562495843e-16, -1.235503676226059e-16, -7.550300243603693e-17, -4.768610680170754e-17],
[ 9.385966990329841e-15, 1.04288522114776e-15, 2.08577044229552e-16, 5.688464842624146e-17, 1.896154947541382e-17, 7.292903644389931e-18, 3.125530133309971e-18, 1.458580728877986e-18, 7.292903644389931e-19, 3.860948988206434e-19, 2.144971660114686e-19, 1.241825697961134e-19, 7.450954187766803e-20],
[ -2.896903392077112e-17, -2.896903392077112e-18, -5.267097076503839e-19, -1.31677426912596e-19, -4.051613135772184e-20, -1.447004691347209e-20, -5.788018765388834e-21, -2.532258209857615e-21, -1.191650922285936e-21, -5.958254611429682e-22, -3.135923479699833e-22, -1.724757913834908e-22, -9.855759507628046e-23],
[ 7.242258480192779e-20, 6.583871345629799e-21, 1.0973118909383e-21, 2.532258209857615e-22, 7.235023456736043e-23, 2.411674485578681e-23, 9.043779320920054e-24, 3.723909132143551e-24, 1.655070725397134e-24, 7.839808699249582e-25, 3.919904349624791e-25, 2.053283230755843e-25, 1.119972671321369e-25],
[ -1.496334396734045e-22, -1.246945330611704e-23, -1.918377431710314e-24, -4.110808782236388e-25, -1.096215675263037e-25, -3.42567398519699e-26, -1.209061406540114e-26, -4.701905469878222e-27, -1.979749671527672e-27, -8.908873521874525e-28, -4.242320724702155e-28, -2.121160362351077e-28, -1.10669236296578e-28], 
[ 2.597802772107717e-25, 1.998309824698244e-26, 2.854728320997492e-27, 5.709456641994983e-28, 1.427364160498746e-28, 4.198129883819841e-29, 1.399376627939947e-29, 5.155598102936646e-30, 2.062239241174659e-30, 8.838168176462822e-31, 4.017349171119465e-31, 1.921340907926701e-31, 9.606704539633503e-32]])

assert SERIES_FACTORS.shape == (13,13)

nu_map, m_map = np.meshgrid( np.arange(DERIV_3D_MAX//2), np.arange(DERIV_3D_MAX), indexing='ij')

@traceon_jit
def radial_series_coefficients_3d(v1, v2, v3, z0):
    v1x, v1y, v1z = v1
    v2x, v2y, v2z = v2
    v3x, v3y, v3z = v3
     
    A_sum = np.zeros( (DERIV_3D_MAX//2, DERIV_3D_MAX) )
    B_sum = np.zeros( (DERIV_3D_MAX//2, DERIV_3D_MAX) )
    
    dtheta = thetas[1]-thetas[0]
    
    for b1_, b2_, w in zip(QUAD_B1, QUAD_B2, QUAD_WEIGHTS):
        # Consider every quadrature point on the triangle effectively as a point charge.
        x = v1x + b1_*(v2x-v1x) + b2_*(v3x-v1x)
        y = v1y + b1_*(v2y-v1y) + b2_*(v3y-v1y)
        z = v1z + b1_*(v2z-v1z) + b2_*(v3z-v1z)
        
        r = sqrt(x**2 + y**2 + (z-z0)**2) 
        theta = atan2((z-z0), sqrt(x**2 + y**2))
        mu = atan2(y, x)
         
        # TODO: will this work for points exactly on the z-axis?
        index = int( (theta-thetas[0])/dtheta )
        assert 0 <= index < len(thetas)
        
        t = theta-thetas[index]
         
        C = thetas_interpolation_coefficients
        A_coeff_base = t**3*C[0,index] + t**2*C[1,index] + t*C[2,index] + C[3,index]
         
        r_dependence = r**(-2*nu_map - m_map - 1)
        
        A_sum += w*A_coeff_base*np.cos(m_map*mu)*r_dependence
        B_sum += w*A_coeff_base*np.sin(m_map*mu)*r_dependence
    
    #A = 1/2*np.linalg.norm(np.cross(v2-v1, v3-v1))
    area = 1/2*sqrt(((v2y-v1y)*(v3z-v1z)-(v2z-v1z)*(v3y-v1y))**2+((v2z-v1z)*(v3x-v1x)-(v2x-v1x)*(v3z-v1z))**2+((v2x-v1x)*(v3y-v1y)-(v2y-v1y)*(v3x-v1x))**2)
    
    return area*A_sum, area*B_sum

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
            sum_ += SERIES_FACTORS[nu, m] * (A[nu, m]*cos(m*phi) + B[nu, m]*sin(m*phi)) * r**(m+2*nu)
        
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
             
            diff_r = SERIES_FACTORS[nu, m] * (A[nu, m]*cos(m*phi) + B[nu, m]*sin(m*phi)) * exp*r**(exp-1)
            diff_theta = SERIES_FACTORS[nu, m] * m *(-A[nu, m]*sin(m*phi) + B[nu, m]*cos(m*phi)) * r**exp
             
            Ex -= diff_r * xp/r + diff_theta * -yp/r**2
            Ey -= diff_r * yp/r + diff_theta *  xp/r**2
            Ez -= SERIES_FACTORS[nu, m] * (Adiff[nu, m]*cos(m*phi) + Bdiff[nu, m]*sin(m*phi)) * r**exp
     
    return np.array([Ex, Ey, Ez])
        
    




 






