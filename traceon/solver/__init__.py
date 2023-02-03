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

from ..util import *
from . import radial_symmetry
from . import planar_odd_symmetry
from .. import excitation as E

# Create cache directory
home = path.expanduser('~')
CACHE_DIR = os.path.join(home, '.traceon', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

class Solution:
     
    def __init__(self, excitation, line_points, line_names, charges):
        self.geometry = excitation.geometry
        self.excitation = excitation
        self.line_points = line_points
        self.line_names = line_names
        self.charges = charges

@traceon_jit
def voltage_contrib(r0, z0, r, z):
    return radial_symmetry._zeroth_deriv_z(r0, z0, r, z)

@traceon_jit
def field_dot_normal(r0, z0, r, z, normal):
     
    Er = -radial_symmetry._first_deriv_r(r0, z0, r, z)
    Ez = -radial_symmetry._first_deriv_z(r0, z0, r, z)
    
    return normal[0].item()*Er + normal[1].item()*Ez

@traceon_jit
def _fill_bem_matrix(matrix,
                    line_points,
                    excitation_types,
                    excitation_values,
                    lines_range):
     
    assert len(excitation_types) == len(excitation_values)
    assert len(excitation_values) <= matrix.shape[0]
    assert matrix.shape[0] == matrix.shape[1]
     
    for i in lines_range:
        p1, p2 = line_points[i]
        r0, z0, _ = (p1+p2)/2
        type_ = excitation_types[i]
        
        if type_ == E.ExcitationType.VOLTAGE_FIXED or \
                type_ == E.ExcitationType.VOLTAGE_FUN or \
                type_ == E.ExcitationType.FLOATING_CONDUCTOR:
            
            for j in range(len(line_points)):
                v1, v2 = line_points[j]
                matrix[i, j] = line_integral(r0, z0, v1[0], v1[1], v2[0], v2[1], voltage_contrib)

        elif type_ == E.ExcitationType.DIELECTRIC:
            normal = np.array(get_normal(p1, p2))
            K = excitation_values[i]
            
            for j in range(len(line_points)):
                v1, v2 = line_points[j]
                # This factor is hard to derive. It takes into account that the field
                # calculated at the edge of the dielectric is basically the average of the
                # field at either side of the surface of the dielecric (the field makes a jump).
                matrix[i, j] =  (2*K - 2) / (m.pi*(1 + K)) * line_integral(r0, z0, v1[0], v1[1], v2[0], v2[1], field_dot_normal, args=(normal,))
                 
                if i == j:
                    # When working with dielectrics, the constraint is that
                    # the electric field normal must sum to the surface charge.
                    # The constraint is satisfied by subtracting 1.0 from
                    # the diagonal of the matrix
                    matrix[i, j] -= 1.0

        else:
            raise NotImplementedError('ExcitationType unknown')
        
def _fill_right_hand_side(F, line_points, names,  exc):
    
    for name, indices in names.items():
        type_, value  = exc.excitation_types[name]
         
        if type_ == E.ExcitationType.VOLTAGE_FIXED:
            F[indices] = value
        elif type_ == E.ExcitationType.VOLTAGE_FUN:
            for i in indices:
                F[i] = value(*( (line_points[i][0] + line_points[i][1])/2 ))
        elif type_ == E.ExcitationType.DIELECTRIC or \
                type_ == E.ExcitationType.FLOATING_CONDUCTOR:
            F[indices] = 0
     
    return F

def _add_floating_conductor_constraints(matrix, F, active_lines, active_names, excitation):
    
    floating = [n for n in active_names.keys() if excitation.excitation_types[n][0] == E.ExcitationType.FLOATING_CONDUCTOR]
    N_matrix = matrix.shape[0]
    assert F.size == N_matrix

    for i, f in enumerate(floating):
        for index in active_names[f]:
            # An extra unknown voltage is added to the matrix for every floating conductor.
            # The column related to this unknown voltage is positioned at the rightmost edge of the matrix.
            # If multiple floating conductors are present the column lives at -len(floating) + i
            matrix[ index, -len(floating) + i] = -1
            # The unknown voltage is determined by the constraint on the total charge of the conductor.
            # This constraint lives at the bottom edge of the matrix.
            # The surface area of the respective line element is multiplied by the surface charge (unknown)
            # to arrive at the total specified charge (right hand side).
            line = active_lines[index]
            middle = (line[0] + line[1])/2
            length = np.linalg.norm(line[1] - line[0])
            matrix[ -len(floating) + i, index] = length*2*np.pi*middle[0]
            F[-len(floating)+i] = excitation.excitation_types[f][1]
    
def solve_bem(excitation):
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
     
    line_points, names = excitation.get_active_lines()
    N_floating = sum(1 for v in excitation.excitation_types.values() if v[0] == E.ExcitationType.FLOATING_CONDUCTOR)
     
    N_lines = len(line_points)
    N_matrix = N_lines + N_floating # Every floating conductor adds one constraint
    
    excitation_types = np.zeros(N_lines, dtype=np.uint8)
    excitation_values = np.zeros(N_lines)
    
    for n, indices in names.items():
        excitation_types[indices] = int( excitation.excitation_types[n][0] )
        excitation_values[indices] = excitation.excitation_types[n][1]
    
    assert np.all(excitation_types != 0)
     
    print('Total number of line elements: ', N_lines)
     
    THREADS = 2
    split = np.array_split(np.arange(N_lines), THREADS)
    matrices = [np.zeros((N_matrix, N_matrix)) for _ in range(THREADS)]
     
    threads = [Thread(target=_fill_bem_matrix, args=(m, line_points, excitation_types, excitation_values, r)) for r, m in zip(split, matrices)]
    
    st = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
     
    matrix = np.sum(matrices, axis=0)
    F = np.zeros(N_matrix)
    _fill_right_hand_side(F, line_points, names, excitation)
    _add_floating_conductor_constraints(matrix, F, line_points, names, excitation)
    
    print(f'Time for building matrix: {(time.time()-st)*1000:.3f} ms')
     
    assert np.all(np.isfinite(matrix))
    assert np.all(np.isfinite(F))
    
    st = time.time()
     
    charges = np.linalg.solve(matrix, F)
    
    if N_floating > 0:
        # TODO: do not throw away the calculated floating conductor voltages
        # but instead return them in a new fancy 'Solution' class
        charges = charges[:-N_floating]
     
    print(f'Time for solving matrix: {(time.time()-st)*1000:.3f} ms')
     
    assert np.all(np.isfinite(charges))
     
    return (excitation.geometry.symmetry, line_points, charges, excitation.geometry.get_z_bounds())

@traceon_jit 
def get_all_axial_derivatives_at_point(z, solution):
    
    symmetry, lines, charges, _ = solution

    assert symmetry == 'radial'

    derivs = np.zeros( (9, z.size), dtype=np.float64 )

    for i, z_ in enumerate(z):
        for c, l in zip(charges, lines):
            derivs[:, i] += c*radial_symmetry._get_axial_derivatives(np.array([0.0, z_]), l[0], l[1])
    
    return derivs

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

    symmetry, lines, charges, _ = solution

    assert symmetry == 'radial'
    
    potential = 0.0
     
    for c, (v1, v2) in zip(charges, lines):
        potential += c*line_integral(point[0], point[1], v1[0], v1[1], v2[0], v2[1], radial_symmetry._zeroth_deriv_z)

    return potential

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
    
    symmetry, lines, charges, _ = solution
     
    for c, (v1, v2) in zip(charges, lines):
        E[1] -= c*line_integral(point[0], point[1], v1[0], v1[1], v2[0], v2[1], radial_symmetry._first_deriv_z)
    
    if symmetry == 'radial' and abs(point[0]) < 1e-7: # Too close to singularity
        return E
     
    for c, (v1, v2) in zip(charges, lines):
        E[0] -= c*line_integral(point[0], point[1], v1[0], v1[1], v2[0], v2[1], radial_symmetry._first_deriv_r)

    return E


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
        i = min(np.int32( (zp-z0) / dz ), c.shape[2]-1)
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
    _, _, _, (zmin, zmax) = solution
    
    z = np.linspace(zmin, zmax, round( (zmax-zmin)*150)) # 150 points per mm
    dz = z[1] - z[0]
     
    st = time.time()
    
    derivs = get_all_axial_derivatives_at_point(z, solution)
     
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
        if not (z0 < z < zlast):
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
    _, _, _, (zmin, zmax) = solution
     
    @nb.njit
    def f_bem(r, z):
        return field_at_point(np.array([r, z]), solution, zmin=zmin, zmax=zmax)
     
    return f_bem

def _hash_solution(mesh, voltage_dict):
    m = hashlib.sha1()
    m.update(mesh.points)
    m.update(mesh.cells_dict['line'].view(np.uint8))
    
    for k, v in mesh.cell_sets_dict.items():
        m.update(bytes(k, 'utf8'))
        m.update(v['line'].view(np.uint8))
    
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

def field_function_derivs(excitation, recompute=False):
    """Create a field function for the given mesh while the given voltages are applied. The field
    function will use a series expansion in terms of the derivatives of the potential at the optical axis.
    The cache will be checked for the solution, and if not present the solve_bem function will be used.
    
    Args:
        mesh: [meshio](https://github.com/nschloe/meshio) object containing the mesh
        recompute: ignore the cache and always compute the solution using solve_bem. The solution will still
            be saved to the cache afterwards.
        **voltages: the voltages applied on the electrodes.
    """
    geometry = excitation.geometry
    # TODO: generalize cache functionality
    voltages = dict((n, v[1]) for n, v in excitation.excitation_types.items() if v[0] == E.ExcitationType.VOLTAGE_FIXED)
    fn = _cache_filename(geometry.mesh, **voltages)

    assert geometry.symmetry == 'radial'
     
    if not recompute and path.isfile(fn):
        cached = np.load(fn)
        lines, charges, z, derivs = cached['lines'], cached['charges'], cached['z'], cached['derivs']
        solution = ('radial', lines, charges, (z[0],z[-1]))
    else:
        print('Computing BEM solution and saving for voltages: ', voltages)
        solution = solve_bem(excitation)
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
     
    assert geom.symmetry == 'radial'
     
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
















