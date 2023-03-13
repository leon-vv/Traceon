import math as m
import time
from threading import Thread
import os.path as path

import numpy as np
from scipy.interpolate import CubicSpline, BPoly, PPoly
import findiff

from . import geometry as G
from . import excitation as E
from . import backend

FACTOR_AXIAL_DERIV_SAMPLING_2D = 0.2
FACTOR_AXIAL_DERIV_SAMPLING_3D = 0.075

DERIV_ACCURACY = 6

dir_ = path.dirname(__file__)
data = path.join(dir_, 'data')
thetas_file = path.join(data, 'radial-series-3D-thetas.npy')
coefficients_file = path.join(data, 'radial-series-3D-theta-dependent-coefficients.npy')

thetas = np.load(thetas_file)
theta0 = thetas[0]
dtheta = thetas[1]-thetas[0]

thetas_interpolation_coefficients = np.load(coefficients_file)

assert thetas_interpolation_coefficients.shape == (thetas.size-1, backend.DERIV_3D_MAX//2, backend.DERIV_3D_MAX, 4)

def _excitation_to_right_hand_side(excitation, vertices, names):
    floating_names = excitation.get_floating_conductor_names()
     
    N_floating = len(floating_names)
    N_lines = len(vertices)
    N_matrix = N_lines + N_floating # Every floating conductor adds one constraint

    F = np.zeros( (N_matrix) )
     
    for name, indices in names.items():
        type_, value  = excitation.excitation_types[name]
         
        if type_ == E.ExcitationType.VOLTAGE_FIXED:
            F[indices] = value
        elif type_ == E.ExcitationType.VOLTAGE_FUN:
            for i in indices:
                points = vertices[i]
                middle = np.average(points, axis=0)
                F[i] = value(*middle)
        elif type_ == E.ExcitationType.DIELECTRIC or \
                type_ == E.ExcitationType.FLOATING_CONDUCTOR:
            F[indices] = 0
    
    # See comments in _add_floating_conductor_constraints_to_matrix
    for i, f in enumerate(floating_names):
        F[-N_floating+i] = excitation.excitation_types[f][1]
     
    assert np.all(np.isfinite(F))
    return F

def area(symmetry, points):
    
    if symmetry == G.Symmetry.RADIAL:
        middle = np.average(points, axis=0)
        length = np.linalg.norm(points[1] - points[0])
        return length*2*np.pi*middle[0]
    elif symmetry == G.Symmetry.THREE_D:
        v1, v2, v3 = points 
        return 1/2*np.linalg.norm(np.cross(v2-v1, v3-v1))


def _add_floating_conductor_constraints_to_matrix(matrix, vertices, names, excitation):
    floating = excitation.get_floating_conductor_names()
    N_matrix = matrix.shape[0]
    assert matrix.shape == (N_matrix, N_matrix)
     
    for i, f in enumerate(floating):
        for index in names[f]:
            # An extra unknown voltage is added to the matrix for every floating conductor.
            # The column related to this unknown voltage is positioned at the rightmost edge of the matrix.
            # If multiple floating conductors are present the column lives at -len(floating) + i
            matrix[ index, -len(floating) + i] = -1
            # The unknown voltage is determined by the constraint on the total charge of the conductor.
            # This constraint lives at the bottom edge of the matrix.
            # The surface area of the respective line element (or triangle) is multiplied by the surface charge (unknown)
            # to arrive at the total specified charge (right hand side).
            element = vertices[index]
            matrix[ -len(floating) + i, index] = area(excitation.mesh.symmetry, element)

def _excitation_to_matrix(excitation, vertices, names):
    floating_names = excitation.get_floating_conductor_names()
    
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
     
    print(f'Total number of elements: {N_lines}, symmetry: {excitation.mesh.symmetry}')
     
    st = time.time()
    THREADS = 2
    
    matrix = np.zeros( (N_matrix, N_matrix) )
    split = np.array_split(np.arange(N_lines), THREADS)
     
    fill_fun = backend.fill_matrix_radial if excitation.mesh.symmetry != G.Symmetry.THREE_D else backend.fill_matrix_3d
    threads = [Thread(target=fill_fun, args=(matrix, vertices, excitation_types, excitation_values, r[0], r[-1])) for r in split]
     
    for t in threads:
        t.start()
    for t in threads:
        t.join()
     
    assert np.all(np.isfinite(matrix))
    _add_floating_conductor_constraints_to_matrix(matrix, vertices, names, excitation)
    print(f'Time for building matrix: {(time.time()-st)*1000:.0f} ms')
        
    return matrix


def _charges_to_field(excitation, charges, vertices, names):
    floating_names = excitation.get_floating_conductor_names()
    N_floating = len(floating_names)
    assert len(charges) == len(vertices) + N_floating
    
    floating_voltages = {n:charges[-N_floating+i] for i, n in enumerate(floating_names)}
    if N_floating > 0:
        charges = charges[:-N_floating]
     
    assert len(charges) == len(vertices)
      
    field_class = FieldRadialBEM if excitation.mesh.symmetry != G.Symmetry.THREE_D else Field3D_BEM
    return field_class(vertices, charges, floating_voltages=floating_voltages)
    

def solve_bem(excitation, superposition=False):
    
    vertices, names = excitation.get_active_vertices()
     
    if not superposition:
        matrix = _excitation_to_matrix(excitation, vertices, names)
        F = _excitation_to_right_hand_side(excitation, vertices, names)
        
        st = time.time()
        charges = np.linalg.solve(matrix, F)
        assert np.all(np.isfinite(charges))
        print(f'Time for solving matrix: {(time.time()-st)*1000:.0f} ms')

        return _charges_to_field(excitation, charges, vertices, names)
     
    excs = excitation.split_for_superposition()
    superposed_names = excs.keys()
    matrix = _excitation_to_matrix(excitation, vertices, names)
    F = np.array([_excitation_to_right_hand_side(excs[n], vertices, names) for n in superposed_names]).T
    st = time.time()
    charges = np.linalg.solve(matrix, F)
    print(f'Time for solving matrix: {(time.time()-st)*1000:.0f} ms')
    assert np.all(np.isfinite(charges))
    return {n:_charges_to_field(excs[n], charges[:, i], vertices, names) for i, n in enumerate(superposed_names)}


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
    
    return c

class Field:
    def __call__(self, *args):
        return self.field_at_point(np.array(args))

class FieldBEM(Field):
    def __init__(self, vertices, charges, floating_voltages={}):
        assert len(vertices) == len(charges)
        self.vertices = vertices
        self.charges = charges
        self.floating_voltages = floating_voltages
     
    def __add__(self, other):
        if isinstance(other, FieldBEM):
            assert np.array_equal(self.vertices, other.vertices), "Cannot add Field3D_BEM if geometry is unequal."
            assert self.charges.shape == other.charges.shape, "Cannot add Field3D_BEM if charges have not equal shape."
            assert set(self.floating_voltages.keys()) == set(other.floating_voltages.keys())
            floating = {n:self.floating_voltages[n]+other.floating_voltages[n] for n in self.floating_voltages.keys()}
            return self.__class__(self.vertices, self.charges+other.charges, floating)
         
        return NotImpemented
    
    def __sub__(self, other):
        return self.__add__(-other)
     
    def __radd__(self, other):
        return self.__add__(other)
     
    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            floating = {n:v*other for n, v in self.floating_voltages.items()}
            return self.__class__(self.vertices, other*self.charges, floating)
        
        return NotImpemented

    def __neg__(self):
        return -1*self
    
    def __rmul__(self, other):
        return self.__mul__(other)


class FieldRadialBEM(FieldBEM):
    def __init__(self, vertices, charges, floating_voltages={}):
        super().__init__(vertices, charges, floating_voltages)
        assert vertices.shape == (len(charges), 2, 3)
        
    def field_at_point(self, point):
        assert point.shape == (2,) or point.shape == (3,)
         
        if point.shape == (2,):
            point = np.array([point[0], point[1], 0.0])
        
        return backend.field_radial(point, self.vertices, self.charges)
    
    def potential_at_point(self, point):
        assert point.shape == (2,) or point.shape == (3,)
        
        if point.shape == (2,):
            point = np.array([point[0], point[1], 0.0])
        
        return backend.potential_radial(point, self.vertices, self.charges)
     
    def get_axial_potential_derivatives(self, z):
        return backend.axial_derivatives_radial_ring(z, self.vertices, self.charges).T
     
    def axial_derivative_interpolation(self, zmin, zmax):
        assert zmax > zmin
        z = np.linspace(zmin, zmax, int(FACTOR_AXIAL_DERIV_SAMPLING_2D*len(self.vertices)))
        
        st = time.time()
        derivs = self.get_axial_potential_derivatives(z)
        coeffs = _quintic_spline_coefficients(z, derivs)
        print(f'Computing derivative interpolation took {(time.time()-st)*1000:.2f} ms ({len(z)} items)')
        
        return FieldRadialAxial(z, coeffs)

class Field3D_BEM(FieldBEM):
    def __init__(self, vertices, charges, floating_voltages={}):
        super().__init__(vertices, charges, floating_voltages)
        assert vertices.shape == (len(charges), 3, 3)
    
    def field_at_point(self, point):
        assert point.shape == (3,)
        return backend.field_3d(point, self.vertices, self.charges)
     
    def potential_at_point(self, point):
        assert point.shape == (3,)
        return backend.potential_3d(point, self.vertices, self.charges)
    
    def axial_derivative_interpolation(self, zmin, zmax):
        assert zmax > zmin
        z = np.linspace(zmin, zmax, int(FACTOR_AXIAL_DERIV_SAMPLING_3D*len(self.vertices)))
        
        print(f'Number of points on z-axis: {len(z)}')
        st = time.time()
        coeffs = backend.axial_coefficients_3d(self.vertices, self.charges, z, thetas, thetas_interpolation_coefficients)
        interpolated_coeffs = CubicSpline(z, coeffs).c
        interpolated_coeffs = np.moveaxis(interpolated_coeffs, 0, -1)
        interpolated_coeffs = np.require(interpolated_coeffs, requirements=('C_CONTIGUOUS', 'ALIGNED'))
        print(f'Time for calculating radial series expansion coefficients: {(time.time()-st)*1000:.0f} ms ({len(z)} items)')

        return Field3DAxial(z, interpolated_coeffs)

class FieldAxial(Field):
    def __init__(self, z, coeffs):
        assert len(z)-1 == len(coeffs)
        assert z[0] < z[-1], "z values in axial interpolation should be ascending"
        self.z = z
        self.coeffs = coeffs
    
    def __add__(self, other):
        if isinstance(other, FieldAxial):
            assert np.array_equal(self.z, other.z), "Cannot add Field3DAxial if optical axis sampling is different."
            assert self.coeffs.shape == other.coeffs.shape, "Cannot add Field3DAxial if shape of axial coefficients is unequal."
            return self.__class__(self.z, self.coeffs+other.coeffs)
         
        return NotImpemented
     
    def __sub__(self, other):
        return self.__add__(-other)
    
    def __radd__(self, other):
        return self.__add__(other)
     
    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.z, other*self.coeffs)
         
        return NotImpemented

    def __neg__(self):
        return -1*self
    
    def __rmul__(self, other):
        return self.__mul__(other)

                
class FieldRadialAxial(FieldAxial):
    def __init__(self, z, coeffs):
        super().__init__(z, coeffs)
        assert coeffs.shape == (len(z)-1, backend.DERIV_2D_MAX, 6)
    
    def field_at_point(self, point):
        assert point.shape == (2,)
        return backend.field_radial_derivs(point, self.z, self.coeffs)
     
    def potential_at_point(self, point):
        assert point.shape == (2,)
        return backend.potential_radial_derivs(point, self.z, self.coeffs)
    

class Field3DAxial(FieldAxial):
    def __init__(self, z, coeffs):
        super().__init__(z, coeffs)
        assert coeffs.shape == (len(z)-1, 2, backend.NU_MAX, backend.M_MAX, 4)
    
    def __call__(self, point):
        return self.field_at_point(point)

    def field_at_point(self, point):
        assert point.shape == (3,)
        return backend.field_3d_derivs(point, self.z, self.coeffs)
     
    def potential_at_point(self, point):
        assert point.shape == (3,)
        return backend.potential_3d_derivs(point, self.z, self.coeffs)
    

