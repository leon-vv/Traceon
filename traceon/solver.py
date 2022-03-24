import math as m
import time
import ctypes
import hashlib
import os
import os.path as path
from threading import Thread
import threading

import numpy as np
import numba as nb
from numba.extending import get_cython_function_address

from scipy.interpolate import CubicSpline

import findiff

# Create cache directory
home = path.expanduser('~')
CACHE_DIR = os.path.join(home, '.traceon', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Number of widths the field point has to be away
# in the boundary element method in order to use an approximation.
WIDTHS_FAR_AWAY = 20
N_FACTOR = 12

def traceon_jit(*args, **kwargs):
    return nb.njit(*args, nogil=True, fastmath=True, **kwargs)

# Simpson integration rule
@traceon_jit
def _simps(y, dx):
    i = 0
    sum_ = 0.0
    while i < len(y)-2:
        sum_ += y[i] + 4*y[i+1] + y[i+2]
        i += 2

    sum_ *= dx/3
    
    if y.size % 2 == 0: # Even, trapezoid on last rule
        sum_ += dx * 0.5 * (y[-1] + y[-2])
    
    return sum_

@traceon_jit(cache=True)
def _norm(x, y):
    return m.sqrt(x**2 + y**2)

addr = get_cython_function_address("scipy.special.cython_special", "ellipk")
ellipk_fn = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(addr)

addr = get_cython_function_address("scipy.special.cython_special", "ellipe")
ellipe_fn = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(addr)

#@nb.njit(nogil=True)
@nb.vectorize('float64(float64)')
def nb_ellipk(x):
    return ellipk_fn(x)

#@nb.njit(nogil=True)
@nb.vectorize('float64(float64)')
def nb_ellipe(x):
    return ellipe_fn(x)

@traceon_jit
def _build_bem_matrix(matrix, points, lines_range, lines):
    assert points.shape == (len(lines), 2, 3)
    assert matrix.shape == (len(lines), len(lines))
    
    for i in lines_range:
        p1, p2 = points[i]
        r0, z0, _ = (p1+p2)/2
        
        for j in range(len(lines)):
            v1, v2 = points[j]
             
            r, z, _ = (v1+v2)/2
            length = _norm(v1[0]-v2[0], v1[1]-v2[1])
            distance = _norm(r-r0, z-z0)
            
            if distance > WIDTHS_FAR_AWAY*length:
                matrix[i, j] = _zeroth_deriv_z(r0, z0, r, z)*length
            else:
                N = N_FACTOR*WIDTHS_FAR_AWAY
                r = np.linspace(v1[0], v2[0], N)
                z = np.linspace(v1[1], v2[1], N)
                ds = _norm(r[1]-r[0], z[1]-z[0])
                to_integrate_ = _zeroth_deriv_z(r0, z0, r, z)
                matrix[i, j] = _simps(to_integrate_, ds)

def solve_bem(mesh, **voltages):
    """Solve for the charges on every line element given a mesh and the voltages applied on the electrodes.
    
    Args:
        mesh: [meshio](https://github.com/nschloe/meshio) object containing the mesh
        **voltages: dictionary relating the electrode names to the applied voltage (in Volt).

    Returns:
        lines: numpy array of line elements defined by two points. Every point has three coordinates and
            the shape is therefore (N, 2, 3) where N is the number of line elements. Only the line elements
            on which a given voltage was applied are returned. The inactive line elements are filtered out.
        
        charges: numpy array with shape (N,) giving the charges on every line element. The unit used for the charges
            is non-standard. Don't use the charges array directly, instead pass it as an argument to the various
            functions in this solver module.
    """ 
    
    lines = mesh.cells_dict['line']
    inactive = np.full(len(lines), True)
    
    for v in voltages.keys():
        inactive[ mesh.cell_sets_dict[v]['line'] ] = False
     
    active_lines = lines[~inactive]
    N = len(active_lines)
    print('Total number of line elements: ', N)
     
    THREADS = 2
    split = np.array_split(np.arange(N), THREADS)
    matrices = [np.zeros((N, N)) for _ in range(THREADS)]
    points = mesh.points[active_lines]
    threads = [Thread(target=_build_bem_matrix, args=(m, points, line_indices, active_lines)) for line_indices, m in zip(split, matrices)]
    
    st = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
     
    print(f'Time for building matrix: {(time.time()-st)*1000:.3f} ms')
    matrix = np.sum(matrices, axis=0)
     
    F = np.zeros(N)
    
    for name, voltage in voltages.items():
        map_index = np.arange(len(lines)) - np.cumsum(inactive)
        F[ map_index[ mesh.cell_sets_dict[name]['line'] ] ] = voltage
    
    assert np.all(np.isfinite(matrix))
    
    st = time.time()
    charges = np.linalg.solve(matrix, F)
    print(f'Time for solving matrix: {(time.time()-st)*1000:.3f} ms')
    
    assert np.all(np.isfinite(charges))
     
    return points, charges
    


# --------------- High order derivatives
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

@traceon_jit(cache=True)
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


@traceon_jit(cache=True)
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

@traceon_jit(cache=True)
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
def _deriv_z_far_away(v0, v1, v2, N):
    mid = (v1+v2)/2
    r, z = mid[0], mid[1]
    r0, z0 = v0[0], v0[1]
    
    length = _norm(v1[0]-v2[0], v1[1]-v2[1])
    
    if N == -1:
        return _first_deriv_r(r0, z0, r, z)*length
    elif N == 0:
        return _zeroth_deriv_z(r0, z0, r, z)*length
    elif N == 1:
        return _first_deriv_z(r0, z0, r, z)*length
    elif N == 2:
        return _second_deriv_z(r0, z0, r, z)*length
    elif N == 3:
        return _third_deriv_z(r0, z0, r, z)*length
    elif N == 4:
        return _fourth_deriv_z(r0, z0, r, z)*length

@traceon_jit
def _deriv_z_close(v0, v1, v2, Nd):
    N = N_FACTOR*WIDTHS_FAR_AWAY
    r0, z0 = v0[0], v0[1]
    r = np.linspace(v1[0], v2[0], N)
    z = np.linspace(v1[1], v2[1], N)
     
    ds = _norm(r[1]-r[0], z[1]-z[0])
     
    if Nd == -1:
        points = _first_deriv_r(r0, z0, r, z)
    if Nd == 0:
        points = _zeroth_deriv_z(r0, z0, r, z)
    if Nd == 1:
        points = _first_deriv_z(r0, z0, r, z)
    elif Nd == 2:
        points = _second_deriv_z(r0, z0, r, z)
    elif Nd == 3:
        points = _third_deriv_z(r0, z0, r, z)
    elif Nd == 4:
        points = _fourth_deriv_z(r0, z0, r, z)
    
    return _simps(points, ds)
 
@traceon_jit
def _deriv_z(v0, v1, v2, N):
    mid = (v1+v2)/2
    width = _norm(v1[0]-v2[0], v1[1]-v2[1])
    distance = _norm(mid[0]-v0[0], mid[1]-v0[1])
    
    if distance > WIDTHS_FAR_AWAY*width:
        return _deriv_z_far_away(v0, v1, v2, N)
    else:
        return _deriv_z_close(v0, v1, v2, N)

@traceon_jit
def deriv_z_at_point(point, lines, charges, N):
    """Compute the derivative of the electrostatic potential (with respect to z0) at the given point.
    
    Args:
        point: coordinates of the point at which the derivative should be samples
        lines: as returned by solve_bem
        charges: as returned by solve_bem
        N: order of the derivative. 0 will simply give the potential. 1 will give the first derivative, etc.

    """
    d = 0.0
    
    assert -1 <= N <= 4
     
    for c, l in zip(charges, lines):
        d += c * _deriv_z(point, l[0], l[1], N)
    
    return d

@traceon_jit
def potential_at_point(point, lines, charges):
    """Compute the potential at a certain point given line elements and the
    corresponding line charges.

    Args:
        point: numpy array of shape (2,) giving the coordinates of the point.
        lines: line elements as returned from solve_bem function
        charges: charges as returned from the solve_bem function
    Returns:
        float: the value of the potential at the given point.
    """
    return deriv_z_at_point(point, lines, charges, 0)

@traceon_jit
def field_at_point(point, lines, charges, zmin=None, zmax=None):
    """Compute the electric field at a certain point given line elements and the
    corresponding line charges.
    
    Args:
        point: numpy array of shape (2,) giving the coordinates of the point.
        lines: line elements as returned from solve_bem function
        charges: charges as returned from the solve_bem function
    Returns:
        float: the value of the electric field in the r direction (Er)
        float: the value of the electric field in the z direction (Ez)
    """

    E = np.array([0.0, 0.0])
     
    if zmax != None and point[1] > zmax:
        return E
     
    if zmin != None and point[1] < zmin:
        return E
    
    Ez = -deriv_z_at_point(point, lines, charges, 1)
    
    if abs(point[0]) < 1e-7: # Too close to singularity
        return np.array([0.0, Ez])
     
    return np.array([-deriv_z_at_point(point, lines, charges, -1), Ez])


def _interpolate_numba(z, derivs):
    # k is degree of polynomial
    assert derivs.shape == (9, z.size)
    c = np.zeros( (9, 4, z.size-1) )
    
    for i, d in enumerate(derivs):
        ppoly = CubicSpline(z, d)
        c[i, :, :], x, k = ppoly.c, ppoly.x, ppoly.c.shape[0]-1
        assert np.all(x == z)
        assert k == 3
    
    z0, zlast = z[0], z[-1]
    dz = z[1] - z[0]
    
    assert np.all(np.isclose(np.diff(z), dz)), 'Not equidistant point in ppoly'
     
    @nb.njit(fastmath=True)
    def compute(zp):
        i = np.int32(np.floor( (zp-z0) / dz))
        assert 0 <= i < c.shape[2]
        diffz = zp - z[i]

        d1, d2 = diffz, diffz**2
        d3 = d2*diffz
        
        return (
            c[1, 3, i] + c[1, 2, i]*diffz + c[1, 1, i]*d2 + c[1, 0, i]*d3,
            c[2, 3, i] + c[2, 2, i]*diffz + c[2, 1, i]*d2 + c[2, 0, i]*d3,
            c[3, 3, i] + c[3, 2, i]*diffz + c[3, 1, i]*d2 + c[3, 0, i]*d3,
            c[4, 3, i] + c[4, 2, i]*diffz + c[4, 1, i]*d2 + c[4, 0, i]*d3,
            c[5, 3, i] + c[5, 2, i]*diffz + c[5, 1, i]*d2 + c[5, 0, i]*d3,
            c[6, 3, i] + c[6, 2, i]*diffz + c[6, 1, i]*d2 + c[6, 0, i]*d3,
            c[7, 3, i] + c[7, 2, i]*diffz + c[7, 1, i]*d2 + c[7, 0, i]*d3,
            c[8, 3, i] + c[8, 2, i]*diffz + c[8, 1, i]*d2 + c[8, 0, i]*d3)
     
    return compute

def _get_axial_derivatives(lines, charges):
    zmin = np.min(lines[:, :, 1])+0.01#+0.05
    zmax = np.max(lines[:, :, 1])-0.01#-0.25

    z = np.linspace(zmin, zmax, round( (zmax-zmin)*150)) # 150 points per mm
    dz = z[1] - z[0]
     
    st = time.time()
    derivs = [ np.array([deriv_z_at_point(np.array([0.0, z]), lines, charges, i) for z in z]) for i in range(5) ]
      
    derivs.append(findiff.FinDiff(0, dz, 1, acc=6)(derivs[4])) # 5th derivatve
    derivs.append(findiff.FinDiff(0, dz, 2, acc=6)(derivs[4])) # 6th derivative
    derivs.append(findiff.FinDiff(0, dz, 3, acc=6)(derivs[4])) # 7th derivative
    derivs.append(findiff.FinDiff(0, dz, 4, acc=6)(derivs[4])) # 8th derivative
    print(f'Computing derivatives took {(time.time()-st):.2f} s')
     
    assert all(derivs[i].shape == derivs[0].shape for i in range(1, 9))
     
    return z, np.array(derivs)


def _field_from_derivatives(z, derivs):
    # derivs[0] is potentail
    # derivs[1] is first derivative, etc.

    assert derivs.shape[0] == 9
     
    st = time.time()
    z0, zlast = z[0], z[-1]
    inter = _interpolate_numba(z, derivs)
      
    @nb.njit
    def field(r, z):
        if not (z0 <= z < zlast):
            return np.array([0.0, 0.0])
         
        i1, i2, i3, i4, i5, i6, i7, i8 = inter(z)
         
        return np.array([
            r/2*(i2 - r**2/8*i4 + r**4/192*i6 - r**6/9216*i8),
            -i1 + r**2/4*i3 - r**4/64*i5 + r**6/2304*i7])
     
    field(0.01, 1.0)
     
    print(f'Computing derivatives and compilation took {(time.time()-st):.2f} s')
   
    return field

def field_function_bem(lines, charges):
    """Create a field function using the conventional BEM field evaluation.
    Every field evaluation causes an iteration over all the line charges and is
    therefore much slower than the field function returned by 'field_function_derivs'.
    
    Args:
        lines: lines as returned by 'solve_bem'
        charges: charges as returned by 'solve_bem'
    
    Returns:
        Field function (callsign f(r, z) -> [Er, Ez])
    """
    field_zmin = np.min(lines[:, :, 1])
    field_zmax = np.max(lines[:, :, 1])-0.25
     
    @nb.njit
    def f_bem(r, z):
        return field_at_point(np.array([r, z]), lines, charges, zmin=field_zmin, zmax=field_zmax)
     
    return f_bem

def _hash_solution(mesh, voltage_dict):
    m = hashlib.sha1()
    m.update(mesh.cells_dict['line'].view(np.uint8))
    m.update(bytes(str(voltage_dict), 'utf8'))
    return m.hexdigest()
 
def solution_exists_in_cache(mesh, **voltages):
    """Check whether the potential corresponding to the mesh and voltages has already been computed
    and is available in the cache (usually ~/.traceon/cache/)."""
    for k in voltages.keys():
        voltages[k] = float(voltages[k])
     
    hash_ = _hash_solution(mesh, voltages)
    fn = path.join(CACHE_DIR, hash_ + '.npz')
    
    return path.isfile(fn)   
     
def field_function_derivs(mesh, recompute=False, **voltages):
    """Create a field function for the given mesh while the given voltages are applied. The field
    function will use a series expansion in terms of the derivatives of the potential at the optical axis.
    The cache will be checked for the solution, and if not present the solve_bem function will be used.
    
    Args:
        mesh: [meshio](https://github.com/nschloe/meshio) object containing the mesh
        recompute: ignore the cache and always compute the solution using solve_bem. The solution will still
            be saved to the cache afterwards.
        **voltages: the voltages applied on the electrodes.
    """
    for k in voltages.keys():
        voltages[k] = float(voltages[k])
    
    hash_ = _hash_solution(mesh, voltages)
    fn = path.join(CACHE_DIR, hash_ + '.npz')
     
    if not recompute and path.isfile(fn):
        cached = np.load(fn)
        lines, charges, z, derivs = cached['lines'], cached['charges'], cached['z'], cached['derivs']
    else:
        print('Computing BEM solution and saving for voltages: ', voltages)
        lines, charges = solve_bem(mesh, **voltages)
        z, derivs = _get_axial_derivatives(lines, charges)
        np.savez(fn, lines=lines, charges=charges, z=z, derivs=derivs)
     
    return lines, charges, _field_from_derivatives(z, derivs)
   
def field_function_superposition(*funs):
    """Given a number of field functions (currently at most 4) return a field function which is the superposition
    of all the field functions. The returned field function will accept the voltages as auxillary parameters.
    
    Args:
        funs: list of field functions (length at most 4)

    Returns:
        field function: the superposition of the field functions. For example, for two field functions as input
            the callsign of the returned function will be f(r, z, v1, v2) and evaluate to v1*f1(r,z) + v2*f2(r,z)
    """
    
    assert len(funs) in [1,2,3,4], "Superposition currently supported for up to 4 fields"
     
    if len(funs) == 1:
        f1, = funs
        
        @nb.njit
        def field1(r, z, v1):
            return v1*f1(r,z)
        
        return field1
    elif len(funs) == 2:
        f1, f2 = funs 
        
        @nb.njit
        def field2(r, z, v1, v2):
            return v1*f1(r,z) + v2*f2(r, z)
        
        return field2
    elif len(funs) == 3:
        f1, f2, f3 = funs
         
        @nb.njit
        def field3(r, z, v1, v2, v3):
            return v1*f1(r,z) + v2*f2(r, z) + v3*f3(r, z)
        
        return field3 
    elif len(funs) == 4:
        f1, f2, f3, f4 = funs
          
        @nb.njit
        def field4(r, z, v1, v2, v3, v4):
            return v1*f1(r,z) + v2*f2(r, z) + v3*f3(r, z) + v4*f4(r, z)
        
        return field4


def field_function_superposition_mesh(geom, *non_zero):
    """Given a mesh, return the superposition of the field function corresponding to the electrodes in the
    'non_zero' list. All other electrodes (or named line elements) will be fixed to zero voltage.
    
    Args:
        geom: a Geometry object (see geometry.py)
        non_zero: the electrodes on which a voltage will be applied. 
     
    Returns:
        field function: the superposition of the field functions. For example, if two electrode names are
        supplied the returned field functional will have callsign f(r, z, v1, v2)
    """
     
    N_superposition = len(non_zero)
     
    electrodes = geom.get_electrodes()
     
    lines, charges, fields = [], [], []
    
    for nz in non_zero:
        voltages = dict((e, 0.0 if e != nz else 1.0) for e in electrodes)
        l, c, f = field_function_derivs(geom.mesh, **voltages)
        lines.append(l)
        assert np.all(l == lines[0])
        charges.append(c)
        fields.append(f)
     
    return lines, charges, field_function_superposition(*fields)

def benchmark_field_function(f, *args, r=0.05, z=1.5, N=1000):
    """Given a field function, print out how long one field evaluation takes.
    
    Args:
        f: the field function
        *args: auxillary arguments to pass to f (besides r and z coordinates). This
            should usually be some voltages if the field functional is a superposition.
        N: number of times the field function should be executed
    Returns:
        the time per field evaluation in seconds.
    """
     
    @nb.njit
    def comp(r, z):
        result = np.zeros((r.size, 2))
        for i in range(len(r)):
            result[i, :] = f(r[i], z[i], *args)
         
        return result
    
    # Compile
    comp(np.array([0.1, 0.1]), np.array([0.1, 0.2]))
    r_ = np.full(N, r)
    z_ = np.full(N, z)
    st = time.time()
    comp(r_, z_)[-1]
    end = time.time()
    
    print(f'Field evaluation takes {(end-st)/N*1e6:.2f} us')
    return (end-st)/N
















