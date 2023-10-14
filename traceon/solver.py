"""The solver module uses the Boundary Element Method (BEM) to compute the surface charge distribution of a given
geometry and excitation. Once the surface charge distribution is known, the field at any arbitrary position in space
can be calculated by integration over the charged boundary. However, doing a field evaluation in this manner is very slow
as for every field evaluation an iteration needs to be done over all elements in the mesh. Especially for particle tracing it
is crucial that the field evaluation can be done faster. To achieve this, interpolation techniques can be used. 

The solver package offers interpolation in the form of _radial series expansions_ to drastically increase the speed of ray tracing. For
this consider the `axial_derivative_interpolation` methods documented below.

## Radial series expansion in cylindrical symmetry

Let \( \phi_0(z) \) be the potential along the optical axis. We can express the potential around the optical axis as:

$$
\phi = \phi_0(z_0) - r^2 \\frac{\\partial \phi_0^2}{\\partial z^2} + \\frac{r^4}{64} \\frac{\\partial^4 \phi_0}{\\partial z^4} - \\frac{r^6}{2304} \\frac{\\partial \phi_0^6}{\\partial z^6} + \\cdots
$$

Therefore, if we can efficiently compute the axial potential derivatives \( \\frac{\\partial \phi_0^n}{\\partial z^n} \) we can compute the potential and therefore the fields around the optical axis.
For the derivatives of \( \phi_0(z) \) closed form formulas exist in the case of radially symmetric geometries, see for example formula 13.16a in [1]. Traceon uses a recursive version of these formulas to
very efficiently compute the axial derivatives of the potential.

## Radial series expansion in 3D

In a general three dimensional geometry the potential will be dependent not only on the distance from the optical axis but also on the angle \( \\theta \) around the optical axis
at which the potential is sampled. It turns out (equation (35, 24) in [2]) the potential can be written as follows:

$$
\phi = \sum_{\\nu=0}^\infty \sum_{m=0}^\infty r^{2\\nu + m} \\left( A^\\nu_m \cos(m\\theta) + B^\\nu_m \sin(m\\theta) \\right)
$$

The \(A^\\nu_m\) and \(B^\\nu_m\) coefficients can be expressed in _directional derivatives_ perpendicular to the optical axis, analogous to the radial symmetric case. The 
mathematics of calculating these coefficients quickly and accurately gets quite involved, but all details have been abstracted away from the user.

### References
[1] P. Hawkes, E. Kasper. Principles of Electron Optics. Volume one: Basic Geometrical Optics. 2018.

[2] W. Glaser. Grundlagen der Elektronenoptik. 1952.

"""


import math as m
import time
from threading import Thread
import os.path as path

import numpy as np
from scipy.interpolate import CubicSpline, BPoly, PPoly
from scipy.special import legendre
from scipy.integrate import quad

from . import geometry as G
from . import excitation as E
from . import backend
from . import util
from . import fast_multipole_method

FACTOR_AXIAL_DERIV_SAMPLING_2D = 0.2
FACTOR_AXIAL_DERIV_SAMPLING_3D = 0.06

DERIV_ACCURACY = 6

def _get_floating_conductor_names(exc):
    return [n for n, (t, v) in exc.excitation_types.items() if t == E.ExcitationType.FLOATING_CONDUCTOR]

def _excitation_to_right_hand_side(excitation, vertices, names):
    floating_names = _get_floating_conductor_names(excitation)
     
    N_floating = len(floating_names)
    N_lines = len(vertices)
    N_matrix = N_lines + N_floating # Every floating conductor adds one constraint

    F = np.zeros( (N_matrix) )
     
    for name, indices in names.items():
        type_, value  = excitation.excitation_types[name]
         
        if type_ == E.ExcitationType.VOLTAGE_FIXED:
            F[indices] = value
        elif type_ == E.ExcitationType.VOLTAGE_FUN:
            positions = [backend.position_and_jacobian_radial(0, vertices[i, 0], vertices[i, 2], vertices[i, 3], vertices[i, 1])[1] for i in indices]
            F[indices] = [value(*p) for p in positions]
        elif type_ == E.ExcitationType.DIELECTRIC or \
                type_ == E.ExcitationType.FLOATING_CONDUCTOR:
            F[indices] = 0
    
    # See comments in _add_floating_conductor_constraints_to_matrix
    for i, f in enumerate(floating_names):
        F[-N_floating+i] = excitation.excitation_types[f][1]
     
    assert np.all(np.isfinite(F))
    return F

def _area(symmetry, jacobian_buffer, pos_buffer, index):
    if symmetry == G.Symmetry.RADIAL:
        return 2*np.pi*np.sum(jacobian_buffer[index] * pos_buffer[index, :, 0])
    elif symmetry == G.Symmetry.THREE_D_HIGHER_ORDER:
        return np.sum(jacobian_buffer[index])

def _add_floating_conductor_constraints_to_matrix(matrix, jac_buffer, pos_buffer, names, excitation):
    floating = _get_floating_conductor_names(excitation)
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
            matrix[ -len(floating) + i, index] = _area(excitation.mesh.symmetry, jac_buffer, pos_buffer, index)

def _excitation_to_matrix(excitation, vertices, names):
    floating_names = _get_floating_conductor_names(excitation)
    
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
     
    st = time.time()
    matrix = np.zeros( (N_matrix, N_matrix) )
    print(f'Using matrix solver, number of elements: {N_lines}, size of matrix: {N_matrix} ({matrix.nbytes/1e6:.0f} MB), symmetry: {excitation.mesh.symmetry}')

    _3d = excitation.mesh.symmetry == G.Symmetry.THREE_D_HIGHER_ORDER

    jac_buffer, pos_buffer = backend.fill_jacobian_buffer_3d_higher_order(vertices) if _3d else backend.fill_jacobian_buffer_radial(vertices)
    fill_fun = backend.fill_matrix_3d if _3d else backend.fill_matrix_radial
     
    def fill_matrix_rows(rows):
        fill_fun(matrix, vertices, excitation_types, excitation_values, jac_buffer, pos_buffer, rows[0], rows[-1])
     
    util.split_collect(fill_matrix_rows, np.arange(N_lines))    

    # Fill the difficult self voltages
    print(f'Time for building matrix: {(time.time()-st)*1000:.0f} ms')

    assert np.all(np.isfinite(matrix))
    
    _add_floating_conductor_constraints_to_matrix(matrix, jac_buffer, pos_buffer, names, excitation)
        
    return matrix, jac_buffer, pos_buffer


def _charges_to_field(excitation, charges, vertices, names, jac_buffer, pos_buffer):
    floating_names = _get_floating_conductor_names(excitation)
    N_floating = len(floating_names)
     
    assert len(charges) == len(vertices) + N_floating
    
    floating_voltages = {n:charges[-N_floating+i] for i, n in enumerate(floating_names)}
    if N_floating > 0:
        charges = charges[:-N_floating]
     
    assert len(charges) == len(vertices)
    
    field_class = FieldRadialBEM if excitation.mesh.symmetry != G.Symmetry.THREE_D_HIGHER_ORDER else Field3D_BEM
    return field_class(vertices, charges, jac_buffer, pos_buffer, floating_voltages=floating_voltages)

def _solve_fmm(excitation, superposition=False, precision=0):
    assert E.ExcitationType.FLOATING_CONDUCTOR not in [t for t, _ in excitation.excitation_types.values()], 'Floating conductor not yet supported in FMM'
    assert excitation.mesh.symmetry == G.Symmetry.THREE_D, "Fast multipole method is only supported for simple 3D geometries (non higher order triangles)."
    assert isinstance(precision, int) and -2 <= precision <= 5
    
    if superposition:
        excs = excitation._split_for_superposition()
        return {n:_solve_fmm(e, superposition=False) for n, e in excs.items()}
    
    triangles, names = excitation.get_active_elements()
    
    print(f'Using FMM solver, number of elements: {len(triangles)}, symmetry: {excitation.mesh.symmetry}, precision: {precision}')
       
    N = len(triangles)
    assert triangles.shape == (N, 3, 3)
     
    F = _excitation_to_right_hand_side(excitation, triangles, names)
    assert F.shape == (N,)
     
    st = time.time()
    charges, count = fast_multipole_method.solve_iteratively(names, excitation, triangles, F, precision=precision)
    print(f'Time for solving FMM: {(time.time()-st)*1000:.0f} ms (iterations: {count})')
     
    jac_buffer, pos_buffer = backend.fill_jacobian_buffer_3d(triangles)
     
    return Field3D_BEM(triangles, charges, jac_buffer, pos_buffer)

def _solve_matrix(excitation, superposition=False):
    vertices, names = excitation.get_active_elements()
     
    if not superposition:
        matrix, jac_buffer, pos_buffer = _excitation_to_matrix(excitation, vertices, names)
          
        F = _excitation_to_right_hand_side(excitation, vertices, names)
        st = time.time()
        charges = np.linalg.solve(matrix, F)
        assert np.all(np.isfinite(charges))
        print(f'Time for solving matrix: {(time.time()-st)*1000:.0f} ms')
        
        return _charges_to_field(excitation, charges, vertices, names, jac_buffer, pos_buffer)
     
    excs = excitation._split_for_superposition()
    superposed_names = excs.keys()
    matrix, jac_buffer, pos_buffer = _excitation_to_matrix(excitation, vertices, names)
    F = np.array([_excitation_to_right_hand_side(excs[n], vertices, names) for n in superposed_names]).T
    st = time.time()
    charges = np.linalg.solve(matrix, F)
    print(f'Time for solving matrix: {(time.time()-st)*1000:.0f} ms')
    assert np.all(np.isfinite(charges))
    return {n:_charges_to_field(excs[n], charges[:, i], vertices, names, jac_buffer, pos_buffer) for i, n in enumerate(superposed_names)}

def solve_bem(excitation, superposition=False, use_fmm=False, fmm_precision=0):
    """
    Solve for the charges on the surface of the geometry by using the Boundary Element Method (BEM) and taking
    into account the specified `excitation`. 

    Parameters
    ----------
    excitation : traceon.excitation.Excitation
        The excitation that produces the resulting field.
     
    superposition : bool
        When `superposition=True` the function returns multiple fields. Each field corresponds with a unity excitation (1V)
        of a physical group that was previously assigned a non-zero fixed voltage value. This is useful when a geometry needs
        to be analyzed for many different voltage settings. In this case taking a linear superposition of the returned fields
        allows to select a different voltage 'setting' without inducing any computational cost. There is no computational cost
        involved in using `superposition=True` since a direct solver is used which easily allows for multiple right hand sides (the
        matrix does not have to be inverted multiple times). However, some excitations are invalid in the superposition process: floating
        conductor with a non-zero total charge and voltage functions (position dependent voltages).
    
    use_fmm : bool
        Use the fast multipole method to calculate the charge distribution. This method is currently only implemented for 3D geometries without
        higher order elements. This function only works if [pyfmmlib](https://github.com/inducer/pyfmmlib) is installed
        (version 2023.1 or later). The fast multipole method is usually slower for small 3D problems, but scales much better to problems with >10^5
        number of triangles.

    fmm_precision : int
        Precision flag passed to the fast multipole library (see iprec argument in the [official documentation](https://github.com/zgimbutas/fmmlib3d/blob/master/doc/fmm3dpart_manual3.pdf)).
        Usually values -1, 0, 1, 2 will work, choose higher numbers if more precision is desired.
    
    Returns
    -------
    A `FieldRadialBEM` if the geometry (contained in the given `excitation`) is radially symmetric. If the geometry is a generic three
    dimensional geometry `Field3D_BEM` is returned. Alternatively, when `superposition=True` a dictionary is returned, where the keys
    are the physical groups with unity excitation, and the values are the resulting fields.
    """

    if use_fmm:
        assert excitation.mesh.symmetry == G.Symmetry.THREE_D, "FMM solver only supported for simple triangular meshes (geometry.Symmetry.THREE_D)"
        return _solve_fmm(excitation, superposition=superposition, precision=fmm_precision)
    else:
        mesh = excitation.mesh
        assert not mesh.is_3d() or mesh.symmetry == G.Symmetry.THREE_D_HIGHER_ORDER, "Matrix solver only supported for higher order triangular meshes (geometry.Symmetry.THREE_D_HIGHER_ORDER)"
        return _solve_matrix(excitation, superposition=superposition)


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
    """An electrostatic field (resulting from surface charges) as computed from the Boundary Element Method. You should
    not initialize this class yourself, but it is used as a base class for the fields returned by the `solve_bem` function. 
    This base class overloads the +,*,- operators so it is very easy to take a superposition of different fields."""
    
    def __init__(self, vertices, charges, jac_buffer, pos_buffer, floating_voltages={}):
        assert len(vertices) == len(charges)
        self.vertices = vertices
        self.charges = charges
        self.floating_voltages = floating_voltages
        self.jac_buffer = jac_buffer
        self.pos_buffer = pos_buffer
        self.field_bounds = None
    
    def set_bounds(self, bounds):
        self.field_bounds = np.array(bounds)

    def __str__(self):
        name = self.__class__.__name__
        return f'<Traceon {name}, number of elements: {len(self.vertices)}>'
     
    def __add__(self, other):
        if isinstance(other, FieldBEM):
            assert np.array_equal(self.vertices, other.vertices), "Cannot add FieldBEM if geometry is unequal."
            assert np.array_equal(self.jac_buffer, other.jac_buffer), "Cannot add FieldBEM if geometry is unequal"
            assert np.array_equal(self.pos_buffer, other.pos_buffer), "Cannot add FieldBEM if geometry is unequal"
            assert self.charges.shape == other.charges.shape, "Cannot add FieldBEM if charges have not equal shape."
            assert set(self.floating_voltages.keys()) == set(other.floating_voltages.keys())
            
            floating = {n:self.floating_voltages[n]+other.floating_voltages[n] for n in self.floating_voltages.keys()}
            return self.__class__(self.vertices, self.charges+other.charges, self.jac_buffer, self.pos_buffer, floating)
         
        return NotImpemented
    
    def __sub__(self, other):
        return self.__add__(-other)
     
    def __radd__(self, other):
        return self.__add__(other)
     
    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            floating = {n:v*other for n, v in self.floating_voltages.items()}
            return self.__class__(self.vertices, other*self.charges, self.jac_buffer, self.pos_buffer, floating)
         
        return NotImpemented

    def __neg__(self):
        return -1*self
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def area_of_elements(self, indices):
        """Compute the total area of the elements at the given indices.
        
        Parameters
        ------------
        indices: int iterable
            Indices giving which elements to include in the area calculation.

        Returns
        ---------------
        The sum of the area of all elements with the given indices.
        """
        return sum(self.area_on_element(i) for i in indices) 
    
    def charge_on_element(self, i):
        return self.area_of_element(i) * self.charges[i]
    
    def charge_on_elements(self, indices):
        """Compute the sum of the charges present on the elements with the given indices. To
        get the total charge of a physical group use `names['name']` for indices where `names` 
        is returned by `traceon.excitation.Excitation.get_active_elements()`.

        Parameters
        ----------
        indices: (N,) array of int
            indices of the elements contributing to the charge sum. 
         
        Returns
        -------
        The sum of the charge. See the note about units on the front page."""
        return sum(self.charge_on_element(i) for i in indices)



class FieldRadialBEM(FieldBEM):
    """A radially symmetric electrostatic field. The field is a result of the surface charges as computed by the
    `solve_bem` function. See the comments in `FieldBEM`."""
    
    def __init__(self, vertices, charges, jac_buffer, pos_buffer, floating_voltages={}):
        super().__init__(vertices, charges, jac_buffer, pos_buffer, floating_voltages)
        assert vertices.shape == (len(charges), 4, 3)
        assert charges.shape == (len(charges),)
        
    def field_at_point(self, point):
        """
        Compute the electric field, \( \\vec{E} = -\\nabla \phi \)
        
        Parameters
        ----------
        point: (2,) array of float64
            Position at which to compute the field.
        
        Returns
        -------
        Numpy array containing the field strengths (in units of V/mm) in the r and z directions.   
        """
        assert point.shape == (2,) or point.shape == (3,)
        return backend.field_radial(point, self.charges, self.jac_buffer, self.pos_buffer)
     
    def potential_at_point(self, point):
        """
        Compute the potential.

        Parameters
        ----------
        point: (2,) array of float64
            Position at which to compute the field.
        
        Returns
        -------
        Potential as a float value (in units of V).
        """
        assert point.shape == (2,) or point.shape == (3,)
        return backend.potential_radial(point, self.charges, self.jac_buffer, self.pos_buffer)
     
    def get_axial_potential_derivatives(self, z):
        """
        Compute the derivatives of the potential at a point on the optical axis (z-axis). 
         
        Parameters
        ----------
        z : (N,) np.ndarray of float64
            Positions on the optical axis at which to compute the derivatives.
        

        Returns
        ------- 
        Numpy array of shape (N, 9) containing the derivatives. At index i one finds the i-th derivative (so
        at position 0 the potential itself is returned). The highest derivative returned is a 
        constant currently set to 9."""
        return backend.axial_derivatives_radial_ring(z, self.charges, self.jac_buffer, self.pos_buffer)
     
    def axial_derivative_interpolation(self, zmin, zmax, N=None):
        """
        Use a radial series expansion based on the potential derivatives at the optical axis
        to allow very fast field evaluations.
        
        Parameters
        ----------
        zmin : float
            Location on the optical axis where to start sampling the derivatives.
            
        zmax : float
            Location on the optical axis where to stop sampling the derivatives. Any field
            evaluation outside [zmin, zmax] will return a zero field strength.
        N: int, optional
            Number of samples to take on the optical axis, if N=None the amount of samples
            is determined by taking into account the number of elements in the mesh.
            

        Returns
        -------
        `FieldRadialAxial` object allowing fast field evaluations.

        """
        assert zmax > zmin
        N = N if N is not None else int(FACTOR_AXIAL_DERIV_SAMPLING_2D*len(self.vertices))
        z = np.linspace(zmin, zmax, N)
        
        st = time.time()
        derivs = np.concatenate(util.split_collect(self.get_axial_potential_derivatives, z), axis=0)
        coeffs = _quintic_spline_coefficients(z, derivs.T)
        print(f'Computing derivative interpolation took {(time.time()-st)*1000:.2f} ms ({len(z)} items)')
        
        return FieldRadialAxial(z, coeffs)
    
    def area_of_element(self, i):
        return _area(G.Symmetry.RADIAL, self.jac_buffer, self.pos_buffer, i)
    
class Field3D_BEM(FieldBEM):
    """An electrostatic field resulting from a general 3D geometry. The field is a result of the surface charges as computed by the
    `solve_bem` function. See the comments in `FieldBEM`."""
     
    def __init__(self, vertices, charges, jac_buffer, pos_buffer, floating_voltages={}):
        super().__init__(vertices, charges, jac_buffer, pos_buffer, floating_voltages)
    
    def field_at_point(self, point):
        """
        Compute the electric field, \( \\vec{E} = -\\nabla \phi \)
        
        Parameters
        ----------
        point: (3,) array of float64
            Position at which to compute the field.
             
        Returns
        -------
        Numpy array containing the field strengths (in units of V/mm) in the x, y and z directions.
        """
        assert point.shape == (3,)
        return backend.field_3d(point, self.charges, self.jac_buffer, self.pos_buffer)
     
    def potential_at_point(self, point):
        """
        Compute the potential.

        Parameters
        ----------
        point: (3,) array of float64
            Position at which to compute the field.
        
        Returns
        -------
        Potential as a float value (in units of V).
        """
        assert point.shape == (3,)
        return backend.potential_3d(point, self.charges, self.jac_buffer, self.pos_buffer)
    
    def axial_derivative_interpolation(self, zmin, zmax, N=None):
        """
        Use a radial series expansion around the optical axis to allow for very fast field
        evaluations. Constructing the radial series expansion in 3D is much more complicated
        than the radial symmetric case, but all details have been abstracted away from the user.
        
        Parameters
        ----------
        zmin : float
            Location on the optical axis where to start sampling the radial expansion coefficients.
            
        zmax : float
            Location on the optical axis where to stop sampling the radial expansion coefficients. Any field
            evaluation outside [zmin, zmax] will return a zero field strength.
        N: int, optional
            Number of samples to take on the optical axis, if N=None the amount of samples
            is determined by taking into account the number of elements in the mesh.
         
        Returns
        -------
        `Field3DAxial` object allowing fast field evaluations.

        """
        assert zmax > zmin

        N = N if N is not None else int(FACTOR_AXIAL_DERIV_SAMPLING_3D*len(self.vertices))
        z = np.linspace(zmin, zmax, N)
        
        print(f'Number of points on z-axis: {len(z)}')
        st = time.time()
        jac_buffer, pos_buffer = self.jac_buffer, self.pos_buffer
        coeffs = util.split_collect(lambda z: backend.axial_coefficients_3d(self.charges, jac_buffer, pos_buffer,  z), z)
        coeffs = np.concatenate(coeffs, axis=0)
        interpolated_coeffs = CubicSpline(z, coeffs).c
        interpolated_coeffs = np.moveaxis(interpolated_coeffs, 0, -1)
        interpolated_coeffs = np.require(interpolated_coeffs, requirements=('C_CONTIGUOUS', 'ALIGNED'))
        print(f'Time for calculating radial series expansion coefficients: {(time.time()-st)*1000:.0f} ms ({len(z)} items)')

        return Field3DAxial(z, interpolated_coeffs)
    
    def area_of_element(self, i):
        return _area(G.Symmetry.THREE_D_HIGHER_ORDER, self.jac_buffer, self.pos_buffer, i)
    

     

class FieldAxial(Field):
    """An electrostatic field resulting from a radial series expansion around the optical axis. You should
    not initialize this class yourself, but it is used as a base class for the fields returned by the `axial_derivative_interpolation` methods. 
    This base class overloads the +,*,- operators so it is very easy to take a superposition of different fields."""
    
    def __init__(self, z, coeffs):
        assert len(z)-1 == len(coeffs)
        assert z[0] < z[-1], "z values in axial interpolation should be ascending"
        self.z = z
        self.coeffs = coeffs
     
    def __str__(self):
        name = self.__class__.__name__
        return f'<Traceon {name}, zmin={self.z[0]} mm, zmax={self.z[-1]} mm,\n\tNumber of samples on optical axis: {len(self.z)}>'
     
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
    """ """
    def __init__(self, z, coeffs):
        super().__init__(z, coeffs)
        assert coeffs.shape == (len(z)-1, backend.DERIV_2D_MAX, 6)
    
    def field_at_point(self, point):
        """
        Compute the electric field, \( \\vec{E} = -\\nabla \phi \)
        
        Parameters
        ----------
        point: (2,) array of float64
            Position at which to compute the field.
             
        Returns
        -------
        Numpy array containing the field strengths (in units of V/mm) in the r and z directions.
        """
        assert point.shape == (2,)
        return backend.field_radial_derivs(point, self.z, self.coeffs)
     
    def potential_at_point(self, point):
        """
        Compute the potential.

        Parameters
        ----------
        point: (2,) array of float64
            Position at which to compute the potential.
        
        Returns
        -------
        Potential as a float value (in units of V).
        """
        assert point.shape == (2,)
        return backend.potential_radial_derivs(point, self.z, self.coeffs)
    

class Field3DAxial(FieldAxial):
    """Field computed using a radial series expansion around the optical axis (z-axis). See comments at the start of this page.
     """
    
    def __init__(self, z, coeffs):
        super().__init__(z, coeffs)
        assert coeffs.shape == (len(z)-1, 2, backend.NU_MAX, backend.M_MAX, 4)
    
    def __call__(self, point):
        return self.field_at_point(point)

    def field_at_point(self, point):
        """
        Compute the electric field, \( \\vec{E} = -\\nabla \phi \)
        
        Parameters
        ----------
        point: (3,) array of float64
            Position at which to compute the field.
             
        Returns
        -------
        Numpy array containing the field strengths (in units of V/mm) in the x, y and z directions.
        """
        assert point.shape == (3,)
        return backend.field_3d_derivs(point, self.z, self.coeffs)
     
    def potential_at_point(self, point):
        """
        Compute the potential.

        Parameters
        ----------
        point: (3,) array of float64
            Position at which to compute the potential.
        
        Returns
        -------
        Potential as a float value (in units of V).
        """
        assert point.shape == (3,)
        return backend.potential_3d_derivs(point, self.z, self.coeffs)
    

