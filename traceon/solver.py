import math as m
import time
from threading import Thread
import os.path as path

import numpy as np
from scipy.interpolate import CubicSpline, BPoly, PPoly
import findiff

from . import excitation as E
from . import backend

FACTOR_AXIAL_DERIV_SAMPLING_2D = 0.2
FACTOR_AXIAL_DERIV_SAMPLING_3D = 0.075

FACTOR_HERMITE_SAMPLING_2D = 1.5
FACTOR_HERMITE_SAMPLING_3D = 2.0

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
        coeffs = backend.axial_coefficients_3d(self.vertices, self.charges, z, thetas, thetas_interpolation_coefficients)
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
        raise NotImplementedError('Hermite interpolation is obsolete')
        
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




