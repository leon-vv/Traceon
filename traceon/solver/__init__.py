import math as m
import time
from threading import Thread

import numpy as np
import numba as nb
from scipy.interpolate import CubicSpline, BPoly, PPoly
import findiff

from ..util import *
from . import radial_symmetry
from . import three_dimensional
from . import planar_odd_symmetry
from .. import excitation as E
from .. import interpolation
from .. import radial_series_interpolation_3d as radial_3d
from .. import backend

FACTOR_AXIAL_DERIV_SAMPLING_2D = 0.2
FACTOR_AXIAL_DERIV_SAMPLING_3D = 0.075

FACTOR_HERMITE_SAMPLING_2D = 1.5
FACTOR_HERMITE_SAMPLING_3D = 2.0

DERIV_ACCURACY = 6

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
            normal = get_normal_3d(p1, p2, p3)
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
     
    st = time.time()
    THREADS = 2
    
    matrix = np.zeros( (N_matrix, N_matrix) )
    split = np.array_split(np.arange(N_lines), THREADS)
     
    fill_fun = backend.fill_matrix_radial if excitation.geometry.symmetry != '3d' else backend.fill_matrix_3d
    threads = [Thread(target=fill_fun, args=(matrix, vertices, excitation_types, excitation_values, r[0], r[-1])) for r in split]
     
    for t in threads:
        t.start()
    for t in threads:
        t.join()
     
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
        Ex, Ey = 0.0, 0.0
        
        for c, (v1, v2) in zip(charges, vertices):
            Ex -= c*line_integral(point, v1, v2, radial_symmetry._first_deriv_r)
            Ey -= c*line_integral(point, v1, v2, radial_symmetry._first_deriv_z)
        
        return np.array([Ex, Ey])
    elif symmetry == '3d':
        Ex, Ey, Ez = 0.0, 0.0, 0.0
         
        for c, (v1, v2, v3) in zip(charges, vertices):
            Ex -= c*triangle_integral(point, v1, v2, v3, three_dimensional._first_deriv_x)
            Ey -= c*triangle_integral(point, v1, v2, v3, three_dimensional._first_deriv_y)
            Ez -= c*triangle_integral(point, v1, v2, v3, three_dimensional._first_deriv_z)
         
        return np.array([Ex, Ey, Ez])

    raise NotImplementedError('Symmetry not recognized')

@traceon_jit
def _field_at_point_superposition(point, scaling, symmetries, vertices, charges):
      
    E = np.array([0.0, 0.0])
     
    for sc, symm, v, c in zip(scaling, symmetries, vertices, charges):
        E += sc * _field_at_point(point, symmetry, v, c)
    
    return E

def _get_one_dimensional_high_order_ppoly(z, y, dydz, dydz2):
    bpoly = BPoly.from_derivatives(z, np.array([y, dydz, dydz2]).T)
    return PPoly.from_bernstein_basis(bpoly)

def _quintic_spline_coefficients(z, derivs):
    # k is degree of polynomial
    #assert derivs.shape == (z.size, backend.DERIV_2D_MAX)
    c = np.zeros( (z.size-1, 9, 6) )
    
    dz = z[1] - z[0]
    assert np.all(np.isclose(np.diff(z), dz)) # Equally spaced
     
    for i, d in enumerate(derivs):
        high_order = i + 2 < len(derivs)
        
        if high_order:
            ppoly = _get_one_dimensional_high_order_ppoly(z, d, derivs[i+1], derivs[i+2])
            start_index = 0
        else:
            ppoly = CubicSpline(z, d)
            start_index = 2
        
        c[:, i, start_index:], x, k = ppoly.c.T, ppoly.x, ppoly.c.shape[0]-1
        assert np.all(x == z)
        assert (high_order and k == 5) or (not high_order and k == 3)
    
    return z, c

@traceon_jit 
def _get_hermite_field_2d(symmetry, lines, charges, x, y):
    assert symmetry == 'radial'
     
    Ex = np.zeros( (x.size, y.size ) )
    Ey = np.zeros( (x.size, y.size) )
    
    for i, x_ in enumerate(x):
        for j, y_ in enumerate(y):
            field = _field_at_point(np.array([x_, y_]), symmetry, lines, charges)

            Ex[i, j] = field[0]
            Ey[i, j] = field[1]
     
    return Ex, Ey

@traceon_jit 
def _get_hermite_field_3d(symmetry, vertices, charges, x, y, z):
    assert symmetry == '3d'
    
    Ex = np.zeros( (x.size, y.size, z.size) )
    Ey = np.zeros( (x.size, y.size, z.size) )
    Ez = np.zeros( (x.size, y.size, z.size) )
    
    for i, x_ in enumerate(x):
        for j, y_ in enumerate(y):
            for k, z_ in enumerate(z):
                field = _field_at_point(np.array([x_, y_, z_]), symmetry, vertices, charges)
                
                Ex[i, j, k] = field[0]
                Ey[i, j, k] = field[1]
                Ez[i, j, k] = field[2]
     
    return Ex, Ey, Ez


@traceon_jit 
def _get_all_axial_derivatives(symmetry, lines, charges, z):
    assert symmetry == 'radial'
     
    derivs = np.zeros( (backend.DERIV_2D_MAX, z.size), dtype=np.float64 )
    
    for i, z_ in enumerate(z):
        for c, l in zip(charges, lines):
            derivs[:, i] += c*line_integral(np.array([0.0, z_]), l[0], l[1], radial_symmetry._get_all_axial_derivatives)
     
    return derivs


@traceon_jit
def _field_from_interpolated_derivatives(point, z_inter, coeff):

    r, z = point[0], point[1]
     
    assert coeff.shape == (z_inter.size-1, backend.DERIV_2D_MAX, 6)
    
    if not (z_inter[0] < z < z_inter[-1]):
        return np.array([0.0, 0.0])
      
    z0 = z_inter[0]
    dz = z_inter[1] - z_inter[0]
    index = np.int32( (z-z0) / dz )
    diffz = z - z_inter[index]
     
    d1, d2, d3, d4, d5 = diffz, diffz**2, diffz**3, diffz**4, diffz**5
     
    # Interpolated derivatives, 0 is potential, 1 is first derivative etc.
    i0, i1, i2, i3, i4, i5, i6, i7, i8 = coeff[index] @ np.array([d5, d4, d3, d2, d1, 1.0])
     
    return np.array([
        r/2*(i2 - r**2/8*i4 + r**4/192*i6 - r**6/9216*i8),
        -i1 + r**2/4*i3 - r**4/64*i5 + r**6/2304*i7])
     

class Field:
     
    def __init__(self, excitation, vertices, names, charges, floating_voltages=None):
        assert len(vertices) == len(charges)
         
        self.geometry = excitation.geometry
        self.excitation = excitation
        self.vertices = vertices
        self.names = names
        self.charges = charges
        assert len(self.charges) == len(self.vertices)
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
        if self.geometry.symmetry == 'radial':
            if point.shape == (2,):
                point = np.array([point[0], point[1], 0.0])
            return backend.field_radial(point, self.vertices, self.charges)
        elif self.geometry.symmetry == '3d':
            assert point.shape == (3,)
            return backend.field_3d(point, self.vertices, self.charges)
             
        raise ValueError('Symmetry not recognized: ' + self.geometry.symmetry)
     
    def potential_at_point(self, point):
        if self.geometry.symmetry == 'radial':
            if point.shape == (2,):
                point = np.array([point[0], point[1], 0.0])
            return backend.potential_radial(point, self.vertices, self.charges)
        elif self.geometry.symmetry == '3d':
            return backend.potential_3d(point, self.vertices, self.charges)
        
        raise ValueError('Symmetry not recognized: ' + self.geometry.symmetry)
    
    def _get_optical_axis_sampling(self, zmin=None, zmax=None):
        idx = 1 if self.geometry.symmetry != '3d' else 2
        
        if zmin is None:
            zmin = self.geometry.bounds[idx][0]
        if zmax is None:
            zmax = self.geometry.bounds[idx][1]
          
        assert zmax > zmin
        # TODO: determine good factor between mesh size and optical axis sampling
        F = FACTOR_AXIAL_DERIV_SAMPLING_3D if self.geometry.symmetry == '3d' else FACTOR_AXIAL_DERIV_SAMPLING_2D
        return np.linspace(zmin, zmax, int(F*self.excitation.get_number_of_active_vertices()))
     
    def get_axial_potential_derivatives(self, z=None):
        assert self.geometry.symmetry == 'radial'
         
        if z is None:
            z = self._get_optical_axis_sampling()
         
        derivs = backend.axial_derivatives_radial_ring(z, self.vertices, self.charges).T
         
        return z, derivs
    
    def get_radial_series_coeffs_3d(self, z=None):
        assert self.geometry.symmetry == '3d'
         
        if z is None:
            z = self._get_optical_axis_sampling()
         
        print(f'Number of points on z-axis: {len(z)}')
        st = time.time()
        coeffs = backend.axial_coefficients_3d(self.vertices, self.charges, z, radial_3d.thetas, radial_3d.thetas_interpolation_coefficients)
        interpolated_coeffs = CubicSpline(z, coeffs).c
        interpolated_coeffs = np.moveaxis(interpolated_coeffs, 0, -1)
        interpolated_coeffs = np.require(interpolated_coeffs, requirements=('C_CONTIGUOUS', 'ALIGNED'))
        print(f'Time for calculating radial series expansion coefficients: {(time.time()-st)*1000:.0f} ms ({len(z)} items)')
        
        return z, interpolated_coeffs

    
    def get_derivative_interpolation_coeffs(self, z=None):
        assert self.geometry.symmetry == 'radial'
        
        if z is None:
            z = self._get_optical_axis_sampling()
        
        for (z_cache, coeffs) in self._derivs_cache:
            if z.shape == z_cache.shape and np.all(z_cache == z):
                return z, coeffs
         
        st = time.time()
        z, derivs = self.get_axial_potential_derivatives(z)
        z, coeffs = _quintic_spline_coefficients(z, derivs)
        print(f'Computing derivative interpolation took {(time.time()-st)*1000:.2f} ms ({len(z)} items)')
        
        self._derivs_cache.append( (z, coeffs) )
        return z, coeffs
    
    def get_hermite_interpolation_coeffs(self, *args, **kwargs):
        if self.geometry.symmetry == '3d':
            return self.get_hermite_interpolation_coeffs_3d(*args, **kwargs)
        else:
            return self.get_hermite_interpolation_coeffs_2d(*args, **kwargs)
        
    def get_hermite_interpolation_coeffs_3d(self, x=None, y=None, z=None, sampling_factor=FACTOR_HERMITE_SAMPLING_3D):
        st = time.time()
         
        N = round(sampling_factor* (self.excitation.get_number_of_active_vertices())**(1/3))
        print(f'Computing hermite with {N}^3 number of points')
        
        if x is None:
            xmin, xmax = self.geometry.bounds[0]
            x = np.linspace(xmin, xmax, N)
        
        if y is None:
            ymin, ymax = self.geometry.bounds[1]
            y = np.linspace(ymin, ymax, N)

        if z is None:
            zmin, zmax = self.geometry.bounds[2]
            z = np.linspace(zmin, zmax, N)
         
        Ex, Ey, Ez = _get_hermite_field_3d(self.geometry.symmetry, self.vertices, self.charges, x, y, z)
        assert all(arr.shape == (x.size, y.size, z.size) for arr in [Ex, Ey, Ez])
         
        dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
        
        DX =  findiff.FinDiff(0, dx, 1, acc=DERIV_ACCURACY)
        DY =  findiff.FinDiff(1, dy, 1, acc=DERIV_ACCURACY)
        DZ =  findiff.FinDiff(2, dz, 1, acc=DERIV_ACCURACY)
        
        DXX = findiff.FinDiff(0, dx, 2, acc=DERIV_ACCURACY)
        DXY = findiff.FinDiff((0, dx, 1), (1, dy, 1), acc=DERIV_ACCURACY)
        DXZ = findiff.FinDiff((0, dx, 1), (2, dz, 1), acc=DERIV_ACCURACY)
        DYY = findiff.FinDiff(1, dy, 2, acc=DERIV_ACCURACY)
        DYZ = findiff.FinDiff((1, dy, 1), (2, dz, 1), acc=DERIV_ACCURACY)
        DZZ = findiff.FinDiff(2, dz, 2, acc=DERIV_ACCURACY)
        
        derivs = np.zeros( (3, x.size, y.size, z.size, 10) )
        
        for i in range(3):
            field = [Ex, Ey, Ez][i]
            derivs[i, :, :, :, 0] = field
            derivs[i, :, :, :, 1] = DX(field)
            derivs[i, :, :, :, 2] = DY(field)
            derivs[i, :, :, :, 3] = DZ(field)
            derivs[i, :, :, :, 4] = DXX(field)
            derivs[i, :, :, :, 5] = DXY(field)
            derivs[i, :, :, :, 6] = DXZ(field)
            derivs[i, :, :, :, 7] = DYY(field)
            derivs[i, :, :, :, 8] = DYZ(field)
            derivs[i, :, :, :, 9] = DZZ(field)
        
        coeffs_x = interpolation.get_hermite_coeffs_3d(x, y, z, derivs[0])
        coeffs_y = interpolation.get_hermite_coeffs_3d(x, y, z, derivs[1])
        coeffs_z = interpolation.get_hermite_coeffs_3d(x, y, z, derivs[2])

        print(f'Computing hermite interpolation coefficients took {(time.time()-st)*1000:.0f} ms')

        return x, y, z, coeffs_x, coeffs_y, coeffs_z

        
        
    def get_hermite_interpolation_coeffs_2d(self, x=None, y=None, z=None, sampling_factor=FACTOR_HERMITE_SAMPLING_2D):
        st = time.time()
         
        N = round(sampling_factor*m.sqrt(self.excitation.get_number_of_active_vertices()))
        
        if x is None:
            xmin, xmax = self.geometry.bounds[0]
            x = np.linspace(xmin, xmax, N)
        
        if y is None:
            ymin, ymax = self.geometry.bounds[1]
            y = np.linspace(ymin, ymax, N)
         
        Ex, Ey = _get_hermite_field_2d(self.geometry.symmetry, self.vertices, self.charges, x, y)
        assert Ex.shape == Ey.shape
         
        dx, dy = x[1]-x[0], y[1]-y[0]
        
        DX =  findiff.FinDiff(0, dx, 1, acc=DERIV_ACCURACY)
        DY =  findiff.FinDiff(1, dy, 1, acc=DERIV_ACCURACY)
        DXX = findiff.FinDiff(0, dx, 2, acc=DERIV_ACCURACY)
        DXY = findiff.FinDiff((0, dx, 1), (1, dy, 1), acc=DERIV_ACCURACY)
        DYY = findiff.FinDiff(1, dy, 2, acc=DERIV_ACCURACY)

        derivs_x = np.zeros( (*Ex.shape, 6) )
        derivs_x[:, :, 0] = Ex
        derivs_x[:, :, 1] = DX(Ex)
        derivs_x[:, :, 2] = DY(Ex)
        derivs_x[:, :, 3] = DXX(Ex)
        derivs_x[:, :, 4] = DXY(Ex)
        derivs_x[:, :, 5] = DYY(Ex)
        
        derivs_y = np.zeros( (*Ey.shape, 6) )
        derivs_y[:, :, 0] = Ey
        derivs_y[:, :, 1] = DX(Ey)
        derivs_y[:, :, 2] = DY(Ey)
        derivs_y[:, :, 3] = DXX(Ey)
        derivs_y[:, :, 4] = DXY(Ey)
        derivs_y[:, :, 5] = DYY(Ey)

        coeffs_x = interpolation.get_hermite_coeffs_2d(x, y, derivs_x)
        coeffs_y = interpolation.get_hermite_coeffs_2d(x, y, derivs_y)

        print(f'Computing hermite interpolation coefficients took {(time.time()-st)*1000:.0f} ms')

        return x, y, coeffs_x, coeffs_y

    def compute_hermite_interpolated_field(self, x, y, coeffs_x, coeffs_y, x_, y_):
        return interpolation.compute_hermite_field_2d(x, y, coeffs_x, coeffs_y, x_, y_)


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






