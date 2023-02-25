import math as m
import numba as nb
import numpy as np

from numba.core.errors import NumbaExperimentalFeatureWarning

def traceon_jit(*args, **kwargs):
    return nb.njit(*args, boundscheck=True, cache=True, nogil=True, fastmath=True, **kwargs)

# Simpson integration rule
@traceon_jit(inline='always')
def simps(function, target_x, target_y, x, y, args=()):
    assert (len(x)-1)%3 == 0
    assert len(x) == len(y)

    dx = norm(x[1]-x[0], y[1]-y[0])
    
    i = 1
    sum_ = function(target_x, target_y, x[0], y[0], *args)
    while i < len(y)-2:
        yi = function(target_x, target_y, x[i], y[i], *args) 
        yi1 = function(target_x, target_y, x[i+1], y[i+1], *args) 
        yi2 = function(target_x, target_y, x[i+2], y[i+2], *args) 
         
        sum_ += 3*yi + 3*yi1 + 2*yi2
        i += 3
    
    assert i == len(y)
     
    # Last one is counted double in the previous iteration
    sum_ -= function(target_x, target_y, x[i-1], y[i-1], *args) 
     
    sum_ *= dx*3/8
      
    return sum_


@traceon_jit(inline='always')
def line_integral(target, v1, v2, function, args=()):
    
    target_x, target_y = target[0], target[1] 
    source_x1, source_y1 = v1[0], v1[1]
    source_x2, source_y2 = v2[0], v2[1]
     
    middle_x = (source_x2 + source_x1)/2
    middle_y = (source_y1 + source_y2)/2
    length = norm(source_x2 - source_x1, source_y2 - source_y1)
    distance = norm(middle_x - target_x, middle_y - target_y)
     
    if distance > 20*length:
        # Speedup, just consider middle point
        return function(target_x, target_y, middle_x, middle_y, *args) * length
    else:
        N_int = 256
        x = np.linspace(source_x1, source_x2, N_int)
        y = np.linspace(source_y1, source_y2, N_int)
        return simps(function, target_x, target_y, x, y, args=args)

@traceon_jit(inline='always')
def triangle_integral(target, v1, v2, v3, function, args=()):
     
    b1 = (0.124949503233232, 0.437525248383384, 0.437525248383384, 0.797112651860071, 0.797112651860071, 0.165409927389841, 0.165409927389841, 0.037477420750088, 0.037477420750088)
    b2 = (0.437525248383384, 0.124949503233232, 0.437525248383384, 0.165409927389841, 0.037477420750088, 0.797112651860071, 0.037477420750088, 0.797112651860071, 0.165409927389841)
    weights = (0.205950504760887, 0.205950504760887, 0.205950504760887, 0.063691414286223, 0.063691414286223, 0.063691414286223, 0.063691414286223, 0.063691414286223, 0.063691414286223)
     
    v1x, v1y, v1z = v1
    v2x, v2y, v2z = v2
    v3x, v3y, v3z = v3
      
    #A = 1/2*np.linalg.norm(np.cross(v2-v1, v3-v1))
    A = 1/2*m.sqrt(((v2y-v1y)*(v3z-v1z)-(v2z-v1z)*(v3y-v1y))**2+((v2z-v1z)*(v3x-v1x)-(v2x-v1x)*(v3z-v1z))**2+((v2x-v1x)*(v3y-v1y)-(v2y-v1y)*(v3x-v1x))**2)
     
    sum_ = 0.0
    for b1_, b2_, w in zip(b1, b2, weights):
        x = v1x + b1_*(v2x-v1x) + b2_*(v3x-v1x)
        y = v1y + b1_*(v2y-v1y) + b2_*(v3y-v1y)
        z = v1z + b1_*(v2z-v1z) + b2_*(v3z-v1z)
        sum_ += w*function(target[0], target[1], target[2], x, y, z, *args)
     
    return A*sum_


@traceon_jit
def norm(x, y):
    return m.sqrt(x**2 + y**2)

@traceon_jit
def norm3d(x, y, z):
    return m.sqrt(x**2 + y**2 + z**3)

@traceon_jit
def get_normal_2d(p1, p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    
    tangent = x2 - x1, y2 - y1
    normal = tangent[1], -tangent[0]
    length = norm(normal[0], normal[1])
    
    return normal[0]/length, normal[1]/length

@traceon_jit
def get_normal_3d(p1, p2, p3):
    normal = np.cross(p2-p1, p3-p1)
    return normal/np.linalg.norm(normal)

# Chebyshev Approximations for the Complete Elliptic Integrals K and E.
# W. J. Cody. 1965.
#
# Augmented with the tricks shown on the Scipy documentation for ellipe and ellipk.

@traceon_jit
def nb_ellipk(k):
    if k > -1:
        return nb_ellipk_singularity(k)
    
    return nb_ellipk_singularity(1 - 1/(1-k))/m.sqrt(1-k)


@traceon_jit
def nb_ellipk_singularity(k):
    eta = 1 - k
    A = (m.log(4),
        9.65736020516771e-2,
        3.08909633861795e-2,
        1.52618320622534e-2,
        1.25565693543211e-2,
        1.68695685967517e-2,
        1.09423810688623e-2,
        1.40704915496101e-3)
    B = (1/2,
        1.24999998585309e-1,
        7.03114105853296e-2,
        4.87379510945218e-2,
        3.57218443007327e-2,
        2.09857677336790e-2,
        5.81807961871996e-3,
        3.42805719229748e-4)
    
    return A[0] + A[1]*eta + A[2]*eta**2 +A[3]*eta**3 + A[4]*eta**4 + A[5]*eta**5 + A[6]*eta**6 + A[7]*eta**7 + \
            + np.log(1/eta)*(B[0] + B[1]*eta + B[2]*eta**2 +B[3]*eta**3 + B[4]*eta**4 + B[5]*eta**5 + B[6]*eta**6 + B[7]*eta**7)

   
@traceon_jit
def nb_ellipe(k):
    if 0 <= k <= 1:
        return nb_ellipe_01(k)
    else:
        return nb_ellipe_01(k/(k-1))*m.sqrt(1-k)

@traceon_jit
def nb_ellipe_01(k):
    eta = 1 - k
    A = (1,
        4.43147193467733e-1,
        5.68115681053803e-2,
        2.21862206993846e-2,
        1.56847700239786e-2,
        1.92284389022977e-2,
        1.21819481486695e-2,
        1.55618744745296e-3)

    B = (0,
        2.49999998448655e-1,
        9.37488062098189e-2,
        5.84950297066166e-2,
        4.09074821593164e-2,
        2.35091602564984e-2,
        6.45682247315060e-3,
        3.78886487349367e-4)
    
    return A[0] + A[1]*eta + A[2]*eta**2 +A[3]*eta**3 + A[4]*eta**4 + A[5]*eta**5 + A[6]*eta**6 + A[7]*eta**7 + \
            + np.log(1/eta)*(B[0] + B[1]*eta + B[2]*eta**2 +B[3]*eta**3 + B[4]*eta**4 + B[5]*eta**5 + B[6]*eta**6 + B[7]*eta**7)



