import math as m
import numba as nb
import numpy as np

def traceon_jit(*args, **kwargs):
    return nb.njit(*args, cache=True, nogil=True, fastmath=True, **kwargs)

# Simpson integration rule
@traceon_jit
def simps(y, dx):
    assert (len(y)-1)%3 == 0
    
    i = 0
    sum_ = 0.0
    while i < len(y)-3:
        sum_ += y[i] + 3*y[i+1] + 3*y[i+2] + y[i+3]
        i += 3
    
    assert i == len(y)-1
    
    sum_ *= dx*3/8
     
    return sum_

@traceon_jit
def line_integral(
    target_x, target_y,
    source_x1, source_y1,
    source_x2, source_y2,
    function,
    *args):

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
        ds = norm(x[1]-x[0], y[1]-y[0])
         
        to_integrate = np.empty(N_int, dtype=np.float64)
        for i in range(N_int):
            to_integrate[i] = function(target_x, target_y, x[i], y[i], *args)
        
        return simps(to_integrate, ds)


@traceon_jit
def norm(x, y):
    return m.sqrt(x**2 + y**2)

@traceon_jit
def get_normal(p1, p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    
    tangent = x2 - x1, y2 - y1
    normal = tangent[1], -tangent[0]
    length = norm(normal[0], normal[1])
    
    return normal[0]/length, normal[1]/length


# Chebyshev Approximations for the Complete Elliptic Integrals K and E.
# W. J. Cody. 1965.

@traceon_jit
def nb_ellipk(k):
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



