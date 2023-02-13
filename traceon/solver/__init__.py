import math as m
import time
import ctypes
import hashlib
import os
import os.path as path
from threading import Thread
import threading
import copy

import numpy as np
import numba as nb
from scipy.interpolate import CubicSpline
import findiff

from ..util import *
from . import radial_symmetry
from . import three_dimensional
from . import planar_odd_symmetry
from .. import excitation as E

# TODO: determine optimal factor
FACTOR_MESH_SIZE_DERIV_SAMPLING = 4

@traceon_jit
def voltage_contrib_2d(r0, z0, r, z):
    return radial_symmetry._zeroth_deriv_z(r0, z0, r, z)

@traceon_jit
def field_dot_normal_2d(r0, z0, r, z, normal):
     
    Er = -radial_symmetry._first_deriv_r(r0, z0, r, z)
    Ez = -radial_symmetry._first_deriv_z(r0, z0, r, z)
    
    return normal[0].item()*Er + normal[1].item()*Ez

@traceon_jit
def _fill_bem_matrix_2d(matrix,
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
                matrix[i, j] = line_integral(np.array([r0, z0]), v1, v2, voltage_contrib_2d)

        elif type_ == E.ExcitationType.DIELECTRIC:
            normal = np.array(get_normal_2d(p1, p2))
            K = excitation_values[i]
            
            for j in range(len(line_points)):
                v1, v2 = line_points[j]
                # This factor is hard to derive. It takes into account that the field
                # calculated at the edge of the dielectric is basically the average of the
                # field at either side of the surface of the dielecric (the field makes a jump).
                matrix[i, j] =  (2*K - 2) / (m.pi*(1 + K)) * line_integral(np.array([r0, z0]), v1, v2, field_dot_normal_2d, args=(normal,))
                 
                if i == j:
                    # When working with dielectrics, the constraint is that
                    # the electric field normal must sum to the surface charge.
                    # The constraint is satisfied by subtracting 1.0 from
                    # the diagonal of the matrix
                    matrix[i, j] -= 1.0

        else:
            raise NotImplementedError('ExcitationType unknown')

@traceon_jit
def voltage_contrib_3d(x0, y0, z0, x, y, z):
    return three_dimensional._zeroth_deriv(x0, y0, z0, x, y, z)

@traceon_jit
def field_dot_normal_3d(x0, y0, z0, x, y, z, normal):
     
    Ex = -three_dimensional._first_deriv_x(x0, y0, z0, x, y, z)
    Ey = -three_dimensional._first_deriv_y(x0, y0, z0, x, y, z)
    Ez = -three_dimensional._first_deriv_z(x0, y0, z0, x, y, z)
     
    return normal[0].item()*Ex + normal[1].item()*Ey + normal[2].item()*Ez


@traceon_jit
def _fill_bem_matrix_3d(matrix,
                    triangle_points,
                    excitation_types,
                    excitation_values,
                    lines_range):
     
    assert len(excitation_types) == len(excitation_values)
    assert len(excitation_values) <= matrix.shape[0]
    assert matrix.shape[0] == matrix.shape[1]
     
    for i in lines_range:
        p1, p2, p3 = triangle_points[i]
        v0 = (p1+p2+p3)/3
        type_ = excitation_types[i]
        
        if type_ == E.ExcitationType.VOLTAGE_FIXED or \
                type_ == E.ExcitationType.VOLTAGE_FUN or \
                type_ == E.ExcitationType.FLOATING_CONDUCTOR:
            
            for j in range(len(triangle_points)):
                v1, v2, v3 = triangle_points[j]
                matrix[i, j] = triangle_integral(v0, v1, v2, v3, voltage_contrib_3d)
        
        elif type_ == E.ExcitationType.DIELECTRIC:
            normal = get_normal_3d(v1, v2, v3)
            K = excitation_values[i]
            
            for j in range(len(triangle_points)):
                v1, v2, v3 = triangle_points[j]
                # This factor is hard to derive. It takes into account that the field
                # calculated at the edge of the dielectric is basically the average of the
                # field at either side of the surface of the dielecric (the field makes a jump).
                matrix[i, j] =  (2*K - 2) / (m.pi*(1 + K)) * triangle_integral(v0, v1, v2, v3, field_dot_normal_3d, args=(normal,))
                 
                if i == j:
                    # When working with dielectrics, the constraint is that
                    # the electric field normal must sum to the surface charge.
                    # The constraint is satisfied by subtracting 1.0 from
                    # the diagonal of the matrix
                    matrix[i, j] -= 1.0

        else:
            raise NotImplementedError('ExcitationType unknown')
 
        
def _fill_right_hand_side(F, points, names,  exc):
    
    for name, indices in names.items():
        type_, value  = exc.excitation_types[name]
         
        if type_ == E.ExcitationType.VOLTAGE_FIXED:
            F[indices] = value
        elif type_ == E.ExcitationType.VOLTAGE_FUN:
            for i in indices:
                middle = np.average(points[i], axis=0)
                F[i] = value(*middle)
        elif type_ == E.ExcitationType.DIELECTRIC or \
                type_ == E.ExcitationType.FLOATING_CONDUCTOR:
            F[indices] = 0
    
    return F

def area(symmetry, points):
    
    if symmetry == 'radial':
        middle = np.average(points, axis=0)
        length = np.linalg.norm(points[1] - points[0])
        return length*2*np.pi*middle[0]
    elif symmetry == '3d':
        v1, v2, v3 = points 
        return 1/2*np.linalg.norm(np.cross(v2-v1, v3-v1))


def _add_floating_conductor_constraints(matrix, F, active_vertices, active_names, excitation):
    
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
            # The surface area of the respective line element (or triangle) is multiplied by the surface charge (unknown)
            # to arrive at the total specified charge (right hand side).
            element = active_vertices[index]
            matrix[ -len(floating) + i, index] = area(excitation.geometry.symmetry, element)
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
     
    vertices, names = excitation.get_active_vertices()

    floating_names = [n for n in names.keys() if excitation.excitation_types[n][0] == E.ExcitationType.FLOATING_CONDUCTOR]
    N_floating = len(floating_names)
     
    N_lines = len(vertices)
    N_matrix = N_lines + N_floating # Every floating conductor adds one constraint
    
    excitation_types = np.zeros(N_lines, dtype=np.uint8)
    excitation_values = np.zeros(N_lines)
    
    for n, indices in names.items():
        excitation_types[indices] = int( excitation.excitation_types[n][0] )

        if excitation.excitation_types[n][0] == E.ExcitationType.DIELECTRIC:
            excitation_values[indices] = excitation.excitation_types[n][1]
     
    assert np.all(excitation_types != 0)
     
    print(f'Total number of elements: {N_lines}, symmetry: {excitation.geometry.symmetry}')
     
    THREADS = 2
    split = np.array_split(np.arange(N_lines), THREADS)
    matrices = [np.zeros((N_matrix, N_matrix)) for _ in range(THREADS)]
     
    fill_fun = _fill_bem_matrix_2d if excitation.geometry.symmetry != '3d' else _fill_bem_matrix_3d
    threads = [Thread(target=fill_fun, args=(m, vertices, excitation_types, excitation_values, r)) for r, m in zip(split, matrices)]
     
    st = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
     
    matrix = np.sum(matrices, axis=0)
    F = np.zeros(N_matrix)
    _fill_right_hand_side(F, vertices, names, excitation)
    _add_floating_conductor_constraints(matrix, F, vertices, names, excitation)
    
    print(f'Time for building matrix: {(time.time()-st)*1000:.0f} ms')
     
    assert np.all(np.isfinite(matrix))
    assert np.all(np.isfinite(F))
    
    st = time.time()
     
    charges = np.linalg.solve(matrix, F)
    
    floating_voltages = None
    
    if N_floating > 0:
        floating_voltages = {n:charges[-N_floating+i] for i, n in enumerate(floating_names)}
        charges = charges[:-N_floating]
     
    print(f'Time for solving matrix: {(time.time()-st)*1000:.0f} ms')
     
    assert np.all(np.isfinite(charges))
    
    return Field(excitation, vertices, names, charges, floating_voltages=floating_voltages)


@traceon_jit
def _potential_at_point(point, symmetry, vertices, charges):
    """Compute the potential at a certain point given line elements and the
    corresponding line charges.

    Args:
        point: numpy array of shape (2,) giving the coordinates of the point.
        lines: line elements as returned from solve_bem function
        charges: charges as returned from the solve_bem function
    Returns:
        float: the value of the potential at the given point.
    """
    potential = 0.0
     
    if symmetry == 'radial':
        for c, (v1, v2) in zip(charges, vertices):
            potential += c*line_integral(point, v1, v2, radial_symmetry._zeroth_deriv_z)
    elif symmetry == '3d':
        for c, (v1, v2, v3) in zip(charges, vertices):
            potential += c*triangle_integral(point, v1, v2, v3, three_dimensional._zeroth_deriv)
    else:
        raise NotImplementedError('Symmetry not recognized')
     
    return potential

@traceon_jit
def _field_at_point(point, symmetry, vertices, charges):
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
    
    if symmetry == 'radial':
        E = np.array([0.0, 0.0])
        
        for c, (v1, v2) in zip(charges, vertices):
            E[1] -= c*line_integral(point, v1, v2, radial_symmetry._first_deriv_z)
        
        if abs(point[0]) < 1e-7: # Too close to singularity
            return E
         
        for c, (v1, v2) in zip(charges, vertices):
            E[0] -= c*line_integral(point, v1, v2, radial_symmetry._first_deriv_r)
        
        return E
    elif symmetry == '3d':
        E = np.array([0.0, 0.0, 0.0])
         
        for c, (v1, v2, v3) in zip(charges, vertices):
            E[0] -= c*triangle_integral(point, v1, v2, v3, three_dimensional._first_deriv_x)
            E[1] -= c*triangle_integral(point, v1, v2, v3, three_dimensional._first_deriv_y)
            E[2] -= c*triangle_integral(point, v1, v2, v3, three_dimensional._first_deriv_z)
         
        return E

    raise NotImplementedError('Symmetry not recognized')

@traceon_jit
def _field_at_point_superposition(point, scaling, symmetries, vertices, charges):
      
    E = np.array([0.0, 0.0])
     
    for sc, symm, v, c in zip(scaling, symmetries, vertices, charges):
        E += sc * _field_at_point(point, symmetry, v, c)
    
    return E


def _cubic_spline_coefficients(z, derivs):
    # k is degree of polynomial
    assert derivs.shape == (9, z.size)
    c = np.zeros( (z.size-1, 9, 4) )
    
    dz = z[1] - z[0]
    assert np.all(np.isclose(np.diff(z), dz))
     
    for i, d in enumerate(derivs):
        ppoly = CubicSpline(z, d)
        c[:, i, :], x, k = ppoly.c.T, ppoly.x, ppoly.c.shape[0]-1
        assert np.all(x == z)
        assert k == 3
    
    return z, c


@traceon_jit 
def _get_all_axial_derivatives(symmetry, lines, charges, z):
    assert symmetry == 'radial'
     
    derivs = np.zeros( (9, z.size), dtype=np.float64 )
    
    for i, z_ in enumerate(z):
        for c, l in zip(charges, lines):
            derivs[:, i] += c*line_integral(np.array([0.0, z_]), l[0], l[1], radial_symmetry._get_all_axial_derivatives)
     
    return derivs


@traceon_jit
def _field_from_interpolated_derivatives(point, z_inter, coeff):

    r, z = point[0], point[1]
     
    assert coeff.shape == (z_inter.size-1, 9, 4)
    
    if not (z_inter[0] < z < z_inter[-1]):
        return np.array([0.0, 0.0])
      
    z0 = z_inter[0]
    dz = z_inter[1] - z_inter[0]
    index = np.int32( (z-z0) / dz )
    diffz = z - z_inter[index]
     
    d1, d2, d3 = diffz, diffz**2, diffz**3
     
    # Interpolated derivatives, 0 is potential, 1 is first derivative etc.
    i0, i1, i2, i3, i4, i5, i6, i7, i8 = coeff[index] @ np.array([d3, d2, diffz, 1])
    
    return np.array([
        r/2*(i2 - r**2/8*i4 + r**4/192*i6 - r**6/9216*i8),
        -i1 + r**2/4*i3 - r**4/64*i5 + r**6/2304*i7])
     

class Field:
     
    def __init__(self, excitation, vertices, line_names, charges, floating_voltages=None):
        assert len(vertices) == len(charges)
         
        self.geometry = excitation.geometry
        self.excitation = excitation
        self.vertices = vertices
        self.line_names = line_names
        self.charges = charges
        self.floating_voltages = floating_voltages

        self._derivs_cache = []

    def __call__(self, point):
        return self.field_at_point(point)
     
    def __add__(self, other):
        if isinstance(other, Field):
            return FieldSuperposition([self, other])
        if isinstance(other, FieldSuperposition):
            return other.__add__(self)

        raise NotImplementedError('Can only add Field or FieldSuperposition to Field (unrecognized type in +)')

    def __mul__(self, other):
        assert isinstance(other, int) or isinstance(other, float), 'Can only multiply Field by int or float (unrecognized type in *)'
        return FieldSuperposition([self], scales=[float(other)])

    def __rmul__(self, other):
        return self.__mul__(other)
     
    def field_at_point(self, point):
        return _field_at_point(point, self.geometry.symmetry, self.vertices, self.charges)
     
    def potential_at_point(self, point):
        return _potential_at_point(point, self.geometry.symmetry, self.vertices, self.charges)

    def _get_optical_axis_sampling(self, zmin=None, zmax=None):
        assert self.geometry.symmetry == 'radial'
        
        # Sample based on mesh size
        mesh_size = self.geometry.get_mesh_size()
        
        if zmin is None:
            zmin = self.geometry.zmin
        if zmax is None:
            zmax = self.geometry.zmax
          
        assert zmax > zmin
        # TODO: determine good factor between mesh size and optical axis sampling
        return np.linspace(zmin, zmax, 4*int((zmax-zmin)/mesh_size))
    
    def get_axial_potential_derivatives(self, z=None):
        assert self.geometry.symmetry == 'radial'
         
        if z is None:
            z = self._get_optical_axis_sampling()
         
        derivs = _get_all_axial_derivatives(self.geometry.symmetry, self.vertices, self.charges, z)
         
        return z, derivs
    
    def get_derivative_interpolation_coeffs(self, z=None):
        assert self.geometry.symmetry == 'radial'
        
        if z is None:
            z = self._get_optical_axis_sampling()
        
        for (z_cache, coeffs) in self._derivs_cache:
            if z.shape == z_cache.shape and np.all(z_cache == z):
                return z, coeffs
         
        st = time.time()
        z, derivs = self.get_axial_potential_derivatives(z)
        z, coeffs = _cubic_spline_coefficients(z, derivs)
        print(f'Computing derivative interpolation took {(time.time()-st)*1000:.2f} ms ({len(z)} items)')
        
        self._derivs_cache.append( (z, coeffs) )
        return z, coeffs


class FieldSuperposition:
    def __init__(self, fields, scales=None):
        self.fields = fields
        
        if scales is None:
            self.scales = [1.0]*len(fields)
        else:
            self.scales = scales

    def __call__( point ):
        assert len(point) == 2
        return np.sum([s*f(point) for s, f in zip(self.scales, self.fields)], axis=0)
    
    def __add__(self, other):
        if isinstance(other, Field):
            return FieldSuperposition(self.field+[other],self.scales+[1])
        if isinstance(other, FieldSuperposition):
            return FieldSuperposition(self.fields+other.fields,self.scales+other.scales)
        
        raise NotImplementedError('Can only add Field or FieldSuperposition to FieldSuperposition (unrecognized type in +)')

    def __mul__(self, other):
        assert isinstance(other, int) or isinstance(other, float), 'Can only multiply FieldSuperposition by int or float (unrecognized type in *)'
        return FieldSuperposition(self.fields, [other*s for s in self.scales])
    
    def get_derivative_interpolation_coeffs(self, z=None):
         
        if z is None:
            mesh_size = min(f.geometry.get_mesh_size() for f in self.fields)
            zmin = min(f.geometry.zmin for f in self.fields)
            zmax = max(f.geometry.zmax for f in self.fields)
            assert zmax > zmin
            z = np.linspace(zmin, zmax, FACTOR_MESH_SIZE_DERIV_SAMPLING*int((zmax-zmin)/mesh_size))
        
        coeffs = np.sum([s*f.get_derivative_interpolation_coeffs()[1] for s, f in zip(self.scales, self.fields)], axis=0)
        return z, coeffs






