from math import floor

import numba as nb

import numpy as np
import matplotlib.pyplot as plt
import findiff

from .util import traceon_jit

# UNITY_COEFFS represent the coefficients of the polynomials
# which compute to unity (either for their value, or one of their derivatives).
# on one of the endpoints of the interval [0, 1].
#
# The coefficients are encoded as UNITY_COEFFS[DERIVATIVE, ENDPOINT].
# For example,
#   UNITY_COEFFS[0, 0] means 0th derivative, left endpoint (so the polynomial takes the value of 1 left, and 0 right)
#   UNITY_COEFFS[1, 0] means 1th derivative, left endpoint (so the first derivative is 1 on the left, and 0 on the right)
#   UNITY_COEFFS[2, 1] means 2th derivative, right endpoint (so the second derivative is 1 on the right, and 0 on the left)
#
# UNITY_COEFFS can be generated using the code below.
# 
# 
#  from scipy.interpolate import BPoly, PPoly
#
#   def get_coeffs(x, y, dydx, d2ydx):
#       bpoly = BPoly.from_derivatives(x, np.array([y, dydx, d2ydx]).T)
#       return np.flip(PPoly.from_bernstein_basis(bpoly).c[:, 0])
#
#   UNITY_COEFFS = np.zeros( (3, 2, 6) )
#   UNITY_COEFFS[0, 0] = get_coeffs([0, 1], [1, 0], [0, 0], [0, 0])
#   UNITY_COEFFS[0, 1] = get_coeffs([0, 1], [0, 1], [0, 0], [0, 0])
#   UNITY_COEFFS[1, 0] = get_coeffs([0, 1], [0, 0], [1, 0], [0, 0])
#   UNITY_COEFFS[1, 1] = get_coeffs([0, 1], [0, 0], [0, 1], [0, 0])
#   UNITY_COEFFS[2, 0] = get_coeffs([0, 1], [0, 0], [0, 0], [1, 0])
#   UNITY_COEFFS[2, 1] = get_coeffs([0, 1], [0, 0], [0, 0], [0, 1])

UNITY_COEFFS = np.array([
    [[  1. ,   0. ,   0. , -10. ,  15. ,  -6. ],
     [  0. ,   0. ,   0. ,  10. , -15. ,   6. ]],
    [[  0. ,   1. ,   0. ,  -6. ,   8. ,  -3. ],
     [  0. ,   0. ,   0. ,  -4. ,   7. ,  -3. ]],
    [[  0. ,   0. ,   0.5,  -1.5,   1.5,  -0.5],
     [  0. ,   0. ,   0. ,   0.5,  -1. ,   0.5]]])

@traceon_jit
def _get_square_coeffs_2d(dx, dy, V, dVdx, dVdy, dVdx2, dVdxy, dVdy2):
    #assert all(arr.shape == (2, 2) for arr in [V, dVdx, dVdy, dVdx2, dVdy2])
     
    coeffs = np.zeros( (6, 6) )
     
    for i in range(2):
        for j in range(2):
            x_coeffs_d0, x_coeffs_d1, x_coeffs_d2 = UNITY_COEFFS[0, i], UNITY_COEFFS[1, i], UNITY_COEFFS[2, i]
            y_coeffs_d0, y_coeffs_d1, y_coeffs_d2 = UNITY_COEFFS[0, j], UNITY_COEFFS[1, j], UNITY_COEFFS[2, j]

            for k in range(6):
                for l in range(6):
                    coeffs[k, l] += \
                        V[i, j] * x_coeffs_d0[k] * y_coeffs_d0[l] + \
                        dy* dVdy[i,j] * x_coeffs_d0[k] * y_coeffs_d1[l] + \
                        dy**2* dVdy2[i,j] * x_coeffs_d0[k] * y_coeffs_d2[l] + \
                        dx*dy* dVdxy[i,j] * x_coeffs_d1[k] * y_coeffs_d1[l] + \
                        dx* dVdx[i, j] * x_coeffs_d1[k] * y_coeffs_d0[l] + \
                        dx**2 * dVdx2[i, j] * x_coeffs_d2[k] * y_coeffs_d0[l]
             
    return coeffs

@traceon_jit
def get_hermite_coeffs_2d(x, y, derivatives):
    #assert all(arr.shape == (x.size, y.size) for arr in [V, dVdx, dVdy, dVdx2, dVdy2])
    
    assert derivatives.shape == (x.size, y.size, 6)        
    
    # Values in derivatives:
    # 0 is V
    # 1 is dVdx
    # 2 is dVdy
    # 3 is dVdx2
    # 4 is dVdxy
    # 5 is dVdy2
    d = derivatives
    V, dVdx, dVdy, dVdx2, dVdxy, dVdy2 = d[:, :, 0], d[:, :, 1], d[:, :, 2], d[:, :, 3], d[:, :, 4], d[:, :, 5]
    
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    
    coeffs = np.zeros( (x.size-1, y.size-1, 6, 6) )
         
    for i in range(x.size-1):
        for j in range(y.size-1):
            coeffs[i, j] = _get_square_coeffs_2d(dx, dy, V[i:i+2, j:j+2], \
                                             dVdx[i:i+2, j:j+2], \
                                             dVdy[i:i+2, j:j+2], \
                                             dVdx2[i:i+2, j:j+2], \
                                             dVdxy[i:i+2, j:j+2], \
                                             dVdy2[i:i+2, j:j+2])
    
    return coeffs

@traceon_jit
def compute_hermite_field_2d(point, x, y, coeffs_x, coeffs_y):
    x_, y_ = point[0], point[1]
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
        
    i, j = int( (x_-x[0])/dx ), int( (y_-y[0])/dy )
    
    if not (0 <= i < x.size-1 and 0 <= j < y.size-1):
        return np.array([0.0, 0.0])
     
    u = (x_-x[i])/dx
    v = (y_-y[j])/dy
     
    Cx = coeffs_x[i, j]
    Cy = coeffs_y[i, j]
    
    sum_x = 0.0
    sum_y = 0.0
     
    for i in range(6):
        for j in range(6):
            sum_x += Cx[i, j] * u**i * v**j
            sum_y += Cy[i, j] * u**i * v**j
      
    return np.array([sum_x, sum_y])


@traceon_jit
def _get_cube_coeffs_3d(dx, dy, dz, derivs):
    assert derivs.shape == (2, 2, 2, 10)
     
    coeffs = np.zeros( (6, 6, 6) )
     
    for i in range(2):
        for j in range(2):
            for p in range(2):
                x_coeffs_d0, x_coeffs_d1, x_coeffs_d2 = UNITY_COEFFS[0, i], UNITY_COEFFS[1, i], UNITY_COEFFS[2, i]
                y_coeffs_d0, y_coeffs_d1, y_coeffs_d2 = UNITY_COEFFS[0, j], UNITY_COEFFS[1, j], UNITY_COEFFS[2, j]
                z_coeffs_d0, z_coeffs_d1, z_coeffs_d2 = UNITY_COEFFS[0, p], UNITY_COEFFS[1, p], UNITY_COEFFS[2, p]

                for k in range(6):
                    for l in range(6):
                        for m in range(6):
                            coeffs[k, l, m] += \
                                        derivs[i,j,p, 0] * x_coeffs_d0[k] * y_coeffs_d0[l] * z_coeffs_d0[m] + \
                                dx*     derivs[i,j,p, 1] * x_coeffs_d1[k] * y_coeffs_d0[l] * z_coeffs_d0[m] + \
                                dy*     derivs[i,j,p, 2] * x_coeffs_d0[k] * y_coeffs_d1[l] * z_coeffs_d0[m] + \
                                dz*     derivs[i,j,p, 3] * x_coeffs_d0[k] * y_coeffs_d0[l] * z_coeffs_d1[m] + \
                                dx*dx*  derivs[i,j,p, 4] * x_coeffs_d2[k] * y_coeffs_d0[l] * z_coeffs_d0[m] + \
                                dx*dy*  derivs[i,j,p, 5] * x_coeffs_d1[k] * y_coeffs_d1[l] * z_coeffs_d0[m] + \
                                dx*dz*  derivs[i,j,p, 6] * x_coeffs_d1[k] * y_coeffs_d0[l] * z_coeffs_d1[m] + \
                                dy*dy*  derivs[i,j,p, 7] * x_coeffs_d0[k] * y_coeffs_d2[l] * z_coeffs_d0[m] + \
                                dy*dz*  derivs[i,j,p, 8] * x_coeffs_d0[k] * y_coeffs_d1[l] * z_coeffs_d1[m] + \
                                dz*dz*  derivs[i,j,p, 9] * x_coeffs_d0[k] * y_coeffs_d0[l] * z_coeffs_d2[m]

      
    return coeffs


@traceon_jit
def get_hermite_coeffs_3d(x, y, z, derivs):
    assert derivs.shape == (x.size, y.size, z.size, 10)
     
    coeffs = np.zeros( (x.size-1, y.size-1, z.size-1, 6, 6, 6) )
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    dz = z[1]-z[0]
     
    for i in range(x.size-1):
        for j in range(y.size-1):
            for k in range(z.size-1):
                coeffs[i, j, k] = _get_cube_coeffs_3d(dx, dy, dz, derivs[i:i+2, j:j+2, k:k+2])
    
    return coeffs

@traceon_jit
def compute_hermite_field_3d(point, x, y, z, coeffs_x, coeffs_y, coeffs_z):
    x_, y_, z_ = point[0], point[1], point[2]
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
        
    i, j, k = int( (x_-x[0])/dx ), int( (y_-y[0])/dy ), int( (z_-z[0])/dz )
     
    if not (0 <= i < x.size-1 and 0 <= j < y.size-1 and 0 <= k < z.size-1):
        return np.array([0.0, 0.0, 0.0])
      
    u = (x_-x[i])/dx
    v = (y_-y[j])/dy
    w = (z_-z[k])/dz
    
    Cx = coeffs_x[i, j, k]
    Cy = coeffs_y[i, j, k]
    Cz = coeffs_z[i, j, k]
    
    sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
     
    for i in range(6):
        for j in range(6):
            for k in range(6):
                sum_x += Cx[i, j, k] * u**i * v**j * w**k
                sum_y += Cy[i, j, k] * u**i * v**j * w**k
                sum_z += Cz[i, j, k] * u**i * v**j * w**k
     
    return np.array([sum_x, sum_y, sum_z])


if __name__ == '__main__':
    ##### --- VALIDATION

    N = 12
    x = np.linspace(0, 5, N)
    y = np.linspace(0, 5, N)
    z = np.linspace(0, 5, N)

    ## 2D Validation

    xx, yy = np.meshgrid(x, y, indexing='ij')

    V = np.sin(xx)*np.cos(yy)
    dVdx = np.cos(xx)*np.cos(yy)
    dVdy = -np.sin(xx)*np.sin(yy)
    dVdx2 = -V
    dVdxy = -np.cos(xx)*np.sin(yy)
    dVdy2 = -V
    
    coeffs = get_hermite_coeffs_2d(x, y, V, dVdx, dVdy, dVdx2, dVdxy, dVdy2)

    for i, x_ in enumerate(x[:-1]):
        for j, y_ in enumerate(y[:-1]):
            assert np.isclose(compute_hermite_potential_2d(x, y, coeffs, x_, y_), V[i, j])
            assert np.isclose(compute_hermite_interp_2d(x, y, coeffs, x_, y_, 0), V[i, j])
            
            field = compute_hermite_field_2d(x, y, coeffs, x_, y_)
            assert np.isclose(field[0], -dVdx[i, j])
            assert np.isclose(field[1], -dVdy[i, j])
             
            assert np.isclose(compute_hermite_interp_2d(x, y, coeffs, x_, y_, 1), dVdx[i, j])
            assert np.isclose(compute_hermite_interp_2d(x, y, coeffs, x_, y_, 2), dVdy[i, j])
            assert np.isclose(compute_hermite_interp_2d(x, y, coeffs, x_, y_, 3), dVdx2[i, j])
            assert np.isclose(compute_hermite_interp_2d(x, y, coeffs, x_, y_, 4), dVdy2[i, j])

    ## 3D Validation

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    assert zz.shape == (N, N, N)

    V = np.sin(xx)*np.cos(yy)*zz
    dVdx = np.cos(xx)*np.cos(yy)*zz
    dVdy = -np.sin(xx)*np.sin(yy)*zz
    dVdz = np.sin(xx)*np.cos(yy)
    dVdx2 = -V*zz
    dVdy2 = -V*zz
    dVdz2 = np.zeros_like(zz)

    coeffs = get_hermite_coeffs_3d(x, y, z, V, dVdx, dVdy, dVdz, dVdx2, dVdy2, dVdz2)

    for i, x_ in enumerate(x[:-1]):
        for j, y_ in enumerate(y[:-1]):
            for k, z_ in enumerate(z[:-1]):
                assert np.isclose(compute_hermite_potential_3d(x, y, z, coeffs, x_, y_, z_), V[i, j, k])
                assert np.isclose(compute_hermite_interp_3d(x, y, z, coeffs, x_, y_, z_, 0), V[i, j, k])
            
                field = compute_hermite_field_3d(x, y, z, coeffs, x_, y_, z_)
                assert np.isclose(field[0], -dVdx[i, j, k])
                assert np.isclose(field[1], -dVdy[i, j, k])
                assert np.isclose(field[2], -dVdz[i, j, k])

                assert np.isclose(compute_hermite_interp_3d(x, y, z, coeffs, x_, y_, z_, 1), dVdx[i, j, k])
                assert np.isclose(compute_hermite_interp_3d(x, y, z, coeffs, x_, y_, z_, 2), dVdy[i, j, k])
                assert np.isclose(compute_hermite_interp_3d(x, y, z, coeffs, x_, y_, z_, 3), dVdz[i, j, k])
                assert np.isclose(compute_hermite_interp_3d(x, y, z, coeffs, x_, y_, z_, 4), dVdx2[i, j, k])
                assert np.isclose(compute_hermite_interp_3d(x, y, z, coeffs, x_, y_, z_, 5), dVdy2[i, j, k])
                assert np.isclose(compute_hermite_interp_3d(x, y, z, coeffs, x_, y_, z_, 6), dVdz2[i, j, k])


'''
yy, xx = np.meshgrid(x, y)


data = np.sin(xx)*np.cos(yy)
datadx = np.cos(xx)*np.cos(yy)
datady = -np.sin(xx)*np.sin(yy)
data = np.sin(xx)*np.cos(yy)
datadx = np.cos(xx)*np.cos(yy)
datady = -np.sin(xx)*np.sin(yy)
datadx2 = -data
datady2 = -data

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#ax.plot_surface(xx, yy, datadx2)

coeffs = get_2d_coeffs(x, y, data, datadx, datady, datadx2, datady2)

print('Done getting 2D coeffs')

x_oversampled = np.linspace(x[0], x[-1]-1e-5, 10*N)
y_oversampled = np.linspace(y[0], y[-1]-1e-5, 10*N)

data = np.zeros( (x_oversampled.size, y_oversampled.size) )

for i in range(len(x_oversampled)):
    for j in range(len(y_oversampled)):
        data[i, j] = compute_value(x, y, coeffs, x_oversampled[i], y_oversampled[j], 3)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
yy, xx = np.meshgrid(x_oversampled, y_oversampled)

#plt.figure()
#plt.plot(xx[:, 10*N//2], data[:, 10*N//2])

#ax.plot_surface(xx, yy, data)
#ax.plot_surface(xx, yy, np.sin(xx)*np.cos(yy))
ax.plot_surface(xx, yy, np.abs(data + np.sin(xx)*np.cos(yy)))

plt.show()
'''

