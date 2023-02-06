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
from . import planar_odd_symmetry
from .. import excitation as E

# TODO: determine optimal factor
FACTOR_MESH_SIZE_DERIV_SAMPLING = 4

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

    floating_names = [n for n in names.keys() if excitation.excitation_types[n][0] == E.ExcitationType.FLOATING_CONDUCTOR]
    N_floating = len(floating_names)
     
    N_lines = len(line_points)
    N_matrix = N_lines + N_floating # Every floating conductor adds one constraint
    
    excitation_types = np.zeros(N_lines, dtype=np.uint8)
    excitation_values = np.zeros(N_lines)
    
    for n, indices in names.items():
        excitation_types[indices] = int( excitation.excitation_types[n][0] )

        if excitation.excitation_types[n][0] == E.ExcitationType.DIELECTRIC:
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
    
    floating_voltages = None
    
    if N_floating > 0:
        floating_voltages = {n:charges[-N_floating+i] for i, n in enumerate(floating_names)}
        charges = charges[:-N_floating]
     
    print(f'Time for solving matrix: {(time.time()-st)*1000:.3f} ms')
     
    assert np.all(np.isfinite(charges))
    
    return Field(excitation, line_points, names, charges, floating_voltages=floating_voltages)


@traceon_jit
def _potential_at_point(r, z, symmetry, lines, charges):
    """Compute the potential at a certain point given line elements and the
    corresponding line charges.

    Args:
        point: numpy array of shape (2,) giving the coordinates of the point.
        lines: line elements as returned from solve_bem function
        charges: charges as returned from the solve_bem function
    Returns:
        float: the value of the potential at the given point.
    """
    assert symmetry == 'radial'
    
    potential = 0.0
     
    for c, (v1, v2) in zip(charges, lines):
        potential += c*line_integral(r, z, v1[0], v1[1], v2[0], v2[1], radial_symmetry._zeroth_deriv_z)
    
    return potential

@traceon_jit
def _field_at_point(r, z, symmetry, lines, charges):
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
     
    for c, (v1, v2) in zip(charges, lines):
        E[1] -= c*line_integral(r, z, v1[0], v1[1], v2[0], v2[1], radial_symmetry._first_deriv_z)
    
    if symmetry == 'radial' and abs(r) < 1e-7: # Too close to singularity
        return E
     
    for c, (v1, v2) in zip(charges, lines):
        E[0] -= c*line_integral(r, z, v1[0], v1[1], v2[0], v2[1], radial_symmetry._first_deriv_r)

    return E

@traceon_jit
def _field_at_point_superposition(r, z, scaling, symmetries, lines, charges):
      
    E = np.array([0.0, 0.0])
     
    for sc, symm, l, c in zip(scaling, symmetries, lines, charges):
        E += sc * _field_at_point(r, z, symm, l, c)
    
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
            derivs[:, i] += c*radial_symmetry._get_axial_derivatives(np.array([0.0, z_]), l[0], l[1])
     
    return derivs


@traceon_jit
def _field_from_interpolated_derivatives(r, z, z_inter, coeff):
    
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
     
    def __init__(self, excitation, line_points, line_names, charges, floating_voltages=None):
        assert len(line_points) == len(charges)
         
        self.geometry = excitation.geometry
        self.excitation = excitation
        self.line_points = line_points
        self.line_names = line_names
        self.charges = charges
        self.floating_voltages = floating_voltages

        self._derivs_cache = []

    def __call__(self, point):
        return self.field_at_point(point)
     
    def __add__(self, other):
        if isinstance(other, Field):
            return FieldSuperposition(self, other)
        if isinstance(other, FieldSuperposition):
            return other.__add__(self)

        raise NotImplementedError('Can only add Field or FieldSuperposition to Field (unrecognized type in +)')

    def __mult__(self, other):
        assert isinstance(other, int) or isinstance(other, float), 'Can only multiply Field by int or float (unrecognized type in *)'
        return FieldSuperposition(self, scales=[float(other)])
    
    def field_at_point(self, point):
        assert len(point) == 2
        return _field_at_point(point[0], point[1], self.geometry.symmetry, self.line_points, self.charges)
     
    def potential_at_point(self, point):
        assert len(point) == 2 
        return _potential_at_point(point[0], point[1], self.geometry.symmetry, self.line_points, self.charges)

    def _get_optical_axis_sampling(self, zmin=None, zmax=None):
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
        
        if z is None:
            z = self._get_optical_axis_sampling()
         
        derivs = _get_all_axial_derivatives(self.geometry.symmetry, self.line_points, self.charges, z)
         
        return z, derivs

    def get_derivative_interpolation_coeffs(self, z=None):

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
    def __init__(self, *fields, scales=None):
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

    def __mult__(self, other):
        assert isinstance(other, int) or isinstance(other, float), 'Can only multiply FieldSuperposition by int or float (unrecognized type in *)'
        return FieldSuperposition(self.fields, [other*s for s in self.scales])
    
    def get_derivative_interpolation_coeffs(self, z=None):
         
        if z is None:
            mesh_size = min(f.geometry.get_mesh_size() for f in self.fields)
            zmin = min(f.geometry.zmin for f in self.fields)
            zmax = max(f.geometry.zmax for f in self.fields)
            assert zmax > zmin
            z = np.linspace(zmin, zmax, FACTOR_MESH_SIZE_DERIV_SAMPLING*int((zmax-zmin)/mesh_size))
        
        coeffs = np.sum([s*f.get_derivative_interpolation_coeffs()[1] for s, f in zip(self.scaling, self.fields)], axis=0)
        return z, coeffs






