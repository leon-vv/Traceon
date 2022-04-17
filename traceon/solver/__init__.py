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
from scipy.interpolate import CubicSpline
import findiff

from .util import *
from . import radial_symmetry

# Create cache directory
home = path.expanduser('~')
CACHE_DIR = os.path.join(home, '.traceon', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Number of widths the field point has to be away
# in the boundary element method in order to use an approximation.
WIDTHS_FAR_AWAY = 20
N_FACTOR = 12

@traceon_jit
def _build_bem_matrix(potential_fun, matrix, points, lines_range, lines):
    assert points.shape == (len(lines), 2, 3)
    assert matrix.shape == (len(lines), len(lines))
    
    for i in lines_range:
        p1, p2 = points[i]
        r0, z0, _ = (p1+p2)/2
        
        for j in range(len(lines)):
            v1, v2 = points[j]
             
            r, z, _ = (v1+v2)/2
            length = norm(v1[0]-v2[0], v1[1]-v2[1])
            distance = norm(r-r0, z-z0)
            
            if distance > WIDTHS_FAR_AWAY*length:
                matrix[i, j] = potential_fun(r0, z0, r, z)*length
            else:
                N = N_FACTOR*WIDTHS_FAR_AWAY
                r = np.linspace(v1[0], v2[0], N)
                z = np.linspace(v1[1], v2[1], N)
                ds = norm(r[1]-r[0], z[1]-z[0])
                to_integrate_ = potential_fun(r0, z0, r, z)
                matrix[i, j] = simps(to_integrate_, ds)

def solve_bem(geometry, **voltages):
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
    
    mesh = geometry.mesh 
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
    potential_fun = radial_symmetry._zeroth_deriv_z
    threads = [Thread(target=_build_bem_matrix, args=(potential_fun, m, points, line_indices, active_lines)) for line_indices, m in zip(split, matrices)]
    
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
     
    return (geometry.symmetry, points, charges)


@traceon_jit
def deriv_z_at_point(point, solution, N):
    """Compute the derivative of the electrostatic potential (with respect to z0) at the given point.
    
    Args:
        point: coordinates of the point at which the derivative should be samples
        lines: as returned by solve_bem
        charges: as returned by solve_bem
        N: order of the derivative. 0 will simply give the potential. 1 will give the first derivative, etc.

    """
    
    assert -1 <= N <= 4

    symmetry, lines, charges = solution
     
    d = 0.0

    if symmetry == 'radial':
        for c, l in zip(charges, lines):
            d += c * radial_symmetry._deriv_z(point, l[0], l[1], N)
     
    return d

@traceon_jit
def potential_at_point(point, solution):
    """Compute the potential at a certain point given line elements and the
    corresponding line charges.

    Args:
        point: numpy array of shape (2,) giving the coordinates of the point.
        lines: line elements as returned from solve_bem function
        charges: charges as returned from the solve_bem function
    Returns:
        float: the value of the potential at the given point.
    """
    return deriv_z_at_point(point, solution, 0)

@traceon_jit
def field_at_point(point, solution, zmin=None, zmax=None):
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
    
    Ez = -deriv_z_at_point(point, solution, 1)
    
    if abs(point[0]) < 1e-7: # Too close to singularity
        return np.array([0.0, Ez])
     
    return np.array([-deriv_z_at_point(point, solution, -1), Ez])


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

def get_axial_derivatives(solution):
    zmin = np.min(solution[1][:, :, 1])+0.01#+0.05
    zmax = np.max(solution[1][:, :, 1])-0.01#-0.25

    z = np.linspace(zmin, zmax, round( (zmax-zmin)*150)) # 150 points per mm
    dz = z[1] - z[0]
     
    st = time.time()
    derivs = [ np.array([deriv_z_at_point(np.array([0.0, z]), solution, i) for z in z]) for i in range(5) ]
     
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

def field_function_bem(solution):
    """Create a field function using the conventional BEM field evaluation.
    Every field evaluation causes an iteration over all the line charges and is
    therefore much slower than the field function returned by 'field_function_derivs'.
    
    Args:
        lines: lines as returned by 'solve_bem'
        charges: charges as returned by 'solve_bem'
    
    Returns:
        Field function (callsign f(r, z) -> [Er, Ez])
    """
    field_zmin = np.min(solution[1][:, :, 1])
    field_zmax = np.max(solution[1][:, :, 1])-0.25
     
    @nb.njit
    def f_bem(r, z):
        return field_at_point(np.array([r, z]), solution, zmin=field_zmin, zmax=field_zmax)
     
    return f_bem

def _hash_solution(mesh, voltage_dict):
    m = hashlib.sha1()
    m.update(mesh.cells_dict['line'].view(np.uint8))
    m.update(bytes(str(voltage_dict), 'utf8'))
    return m.hexdigest()

def _cache_filename(mesh, **voltages):
    for k in voltages.keys():
        voltages[k] = float(voltages[k])
     
    hash_ = _hash_solution(mesh, voltages)
    return path.join(CACHE_DIR, hash_ + '.npz')
 
def solution_exists_in_cache(geom, **voltages):
    """Check whether the potential corresponding to the geometry and voltages has already been computed
    and is available in the cache (usually ~/.traceon/cache/)."""
    return path.isfile(_cache_filename(geom.mesh, **voltages))

def field_function_derivs(geometry, recompute=False, **voltages):
    """Create a field function for the given mesh while the given voltages are applied. The field
    function will use a series expansion in terms of the derivatives of the potential at the optical axis.
    The cache will be checked for the solution, and if not present the solve_bem function will be used.
    
    Args:
        mesh: [meshio](https://github.com/nschloe/meshio) object containing the mesh
        recompute: ignore the cache and always compute the solution using solve_bem. The solution will still
            be saved to the cache afterwards.
        **voltages: the voltages applied on the electrodes.
    """
    fn = _cache_filename(geometry.mesh, **voltages)

    assert geometry.symmetry == 'radial'
     
    if not recompute and path.isfile(fn):
        cached = np.load(fn)
        lines, charges, z, derivs = cached['lines'], cached['charges'], cached['z'], cached['derivs']
        solution = ('radial', lines, charges)
    else:
        print('Computing BEM solution and saving for voltages: ', voltages)
        solution = solve_bem(geometry, **voltages)
        assert solution[0] == 'radial'
        z, derivs = get_axial_derivatives(solution)
        np.savez(fn, lines=solution[1], charges=solution[2], z=z, derivs=derivs)
     
    return solution, _field_from_derivatives(z, derivs)
   
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
     
    solutions, fields = [], []
    
    for nz in non_zero:
        voltages = dict((e, 0.0 if e != nz else 1.0) for e in electrodes)
        solution, f = field_function_derivs(geom, **voltages)
        solutions.append(solution)
        assert np.all(solution[1] == solutions[-1][1])
        fields.append(f)
     
    return solutions, field_function_superposition(*fields)

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
















