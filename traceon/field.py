"""
## Radial series expansion in cylindrical symmetry

Let \\( \\phi_0(z) \\) be the potential along the optical axis. We can express the potential around the optical axis as:

$$
\\phi = \\phi_0(z_0) - \\frac{r^2}{4} \\frac{\\partial \\phi_0^2}{\\partial z^2} + \\frac{r^4}{64} \\frac{\\partial^4 \\phi_0}{\\partial z^4} - \\frac{r^6}{2304} \\frac{\\partial \\phi_0^6}{\\partial z^6} + \\cdots
$$

Therefore, if we can efficiently compute the axial potential derivatives \\( \\frac{\\partial \\phi_0^n}{\\partial z^n} \\) we can compute the potential and therefore the fields around the optical axis.
For the derivatives of \\( \\phi_0(z) \\) closed form formulas exist in the case of radially symmetric geometries, see for example formula 13.16a in [1]. Traceon uses a recursive version of these formulas to
very efficiently compute the axial derivatives of the potential.

[1] P. Hawkes, E. Kasper. Principles of Electron Optics. Volume one: Basic Geometrical Optics. 2018.
"""

import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import CubicSpline, BPoly, PPoly

from . import tracing as T
from . import excitation as E
from . import util
from . import logging
from . import backend

__pdoc__ = {}
__pdoc__['EffectivePointCharges'] = False
__pdoc__['Field.get_low_level_trace_function'] = False
__pdoc__['FieldRadialBEM.get_low_level_trace_function'] = False
__pdoc__['FieldRadialAxial.get_low_level_trace_function'] = False

def _is_numeric(x):
    if isinstance(x, int) or isinstance(x, float) or isinstance(x, np.generic):
        return True

class EffectivePointCharges:
    def __init__(self, charges, jacobians, positions, directions=None):
        self.charges = np.array(charges, dtype=np.float64)
        self.jacobians = np.array(jacobians, dtype=np.float64)
        self.positions = np.array(positions, dtype=np.float64)
        self.directions = directions # Current elements will have a direction

        N = len(self.charges)
        N_QUAD = self.jacobians.shape[1]
        assert self.charges.shape == (N,) and self.jacobians.shape == (N, N_QUAD)
        assert self.positions.shape == (N, N_QUAD, 3) or self.positions.shape == (N, N_QUAD, 2)
        assert self.directions is None or self.directions.shape == (len(self.charges), N_QUAD, 3)
    
    @staticmethod 
    def empty_2d():
        N_QUAD_2D = backend.N_QUAD_2D
        return EffectivePointCharges(np.empty((0,)), np.empty((0, N_QUAD_2D)), np.empty((0,N_QUAD_2D,2)))

    @staticmethod 
    def empty_3d():
        N_TRIANGLE_QUAD = backend.N_TRIANGLE_QUAD
        return EffectivePointCharges(np.empty((0,)), np.empty((0, N_TRIANGLE_QUAD)), np.empty((0, N_TRIANGLE_QUAD, 3)))
    
    @staticmethod 
    def empty_line_3d():
        N_QUAD_2D = backend.N_QUAD_2D
        return EffectivePointCharges(np.empty((0,)), np.empty((0, N_QUAD_2D)), np.empty((0, N_QUAD_2D, 3)), np.empty((0, N_QUAD_2D, 3)))

    def is_2d(self):
        return self.jacobians.shape[1] == backend.N_QUAD_2D
    
    def is_3d(self):
        return self.jacobians.shape[1] == backend.N_TRIANGLE_QUAD
     
    def __len__(self):
        return len(self.charges)
     
    def __add__(self, other):
        if np.array_equal(self.positions, other.positions) and np.array_equal(self.jacobians, other.jacobians):
            return EffectivePointCharges(self.charges + other.charges, self.jacobians, self.positions)
        else:
            return EffectivePointCharges(
                np.concatenate([self.charges, other.charges]),
                np.concatenate([self.jacobians, other.jacobians]),
                np.concatenate([self.positions, other.positions]))
    
    def __radd__(self, other):
        return self.__add__(other)
     
    def __mul__(self, other):
        if _is_numeric(other):
            return EffectivePointCharges(other*self.charges, self.jacobians, self.positions)
        
        return NotImplemented
    
    def __neg__(self):
        return -1*self
    
    def __sub__(self, other):
        return self.__add__(-1*other)
     
    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        dim = '2D' if self.is_2d() else '3D'
        return f'<EffectivePointCharges {dim}\n' \
               f'\tNumber of charges: {len(self.charges)}\n' \
               f'\tJacobian shape:  {self.jacobians.shape}\n' \
               f'\tPositions shape: {self.positions.shape}>'


class Field(ABC):
    """The abstract `Field` class provides the method definitions that all field classes should implement. Note that
    any child clas of the `Field` class can be passed to `traceon.tracing.Tracer` to trace particles through the field."""

    def field_at_point(self, point):
        """Convenience function for getting the field in the case that the field is purely electrostatic
        or magneotstatic. Automatically picks one of `electrostatic_field_at_point` or `magnetostatic_field_at_point`.
        Throws an exception when the field is both electrostatic and magnetostatic.

        Parameters
        ---------------------
        point: (3,) np.ndarray of float64

        Returns
        --------------------
        (3,) np.ndarray of float64. The electrostatic field \\(\\vec{E}\\) or the magnetostatic field \\(\\vec{H}\\).
        """
        elec, mag = self.is_electrostatic(), self.is_magnetostatic()
        
        if elec and not mag:
            return self.electrostatic_field_at_point(point)
        elif not elec and mag:
            return self.magnetostatic_field_at_point(point)
         
        raise RuntimeError("Cannot use field_at_point when both electric and magnetic fields are present, " \
            "use electrostatic_field_at_point or magnetostatic_potential_at_point")
     
    def potential_at_point(self, point):
        """Convenience function for getting the potential in the case that the field is purely electrostatic
        or magneotstatic. Automatically picks one of `electrostatic_potential_at_point` or `magnetostatic_potential_at_point`.
        Throws an exception when the field is both electrostatic and magnetostatic.
         
        Parameters
        ---------------------
        point: (3,) np.ndarray of float64

        Returns
        --------------------
        float. The electrostatic potential (unit Volt) or magnetostaic scalar potential (unit Ampere)
        """
        elec, mag = self.is_electrostatic(), self.is_magnetostatic()
         
        if elec and not mag:
            return self.electrostatic_potential_at_point(point)
        elif not elec and mag:
            return self.magnetostatic_potential_at_point(point) # type: ignore
         
        raise RuntimeError("Cannot use potential_at_point when both electric and magnetic fields are present, " \
            "use electrostatic_potential_at_point or magnetostatic_potential_at_point")

    @abstractmethod
    def is_electrostatic(self):
        ...
    
    @abstractmethod
    def is_magnetostatic(self):
        ...
    
    @abstractmethod
    def electrostatic_potential_at_point(self, point):
        ...
    
    @abstractmethod
    def magnetostatic_field_at_point(self, point):
        ...
    
    @abstractmethod
    def electrostatic_field_at_point(self, point):
        ...
    
    # Following function can be implemented to
    # get a speedup while tracing. Return a 
    # field function implemented in C and a ctypes
    # argument needed. See the field_fun variable in backend/__init__.py 
    # Note that by default it gives back a Python function, which gives no speedup
    def get_low_level_trace_function(self):
        fun = lambda pos, vel: (self.electrostatic_field_at_point(pos), self.magnetostatic_field_at_point(pos))
        return backend.wrap_field_fun(fun), None
 
class FieldBEM(Field, ABC):
    """An electrostatic field (resulting from surface charges) as computed from the Boundary Element Method. You should
    not initialize this class yourself, but it is used as a base class for the fields returned by the `solve_direct` function. 
    This base class overloads the +,*,- operators so it is very easy to take a superposition of different fields."""
    
    def __init__(self, electrostatic_point_charges, magnetostatic_point_charges, current_point_charges):
        assert all([isinstance(eff, EffectivePointCharges) for eff in [electrostatic_point_charges,
                                                                       magnetostatic_point_charges,
                                                                       current_point_charges]])
        self.electrostatic_point_charges = electrostatic_point_charges
        self.magnetostatic_point_charges = magnetostatic_point_charges
        self.current_point_charges = current_point_charges
        self.field_bounds = None
     
    def set_bounds(self, bounds):
        """Set the field bounds. Outside the field bounds the field always returns zero (i.e. no field). Note
        that even in 2D the field bounds needs to be specified for x,y and z axis. The trajectories in the presence
        of magnetostatic field are in general 3D even in radial symmetric geometries.
        
        Parameters
        -------------------
        bounds: (3, 2) np.ndarray of float64
            The min, max value of x, y, z respectively within the field is still computed.
        """
        self.field_bounds = np.array(bounds, dtype=np.float64)
        assert self.field_bounds.shape == (3,2)
    
    def is_electrostatic(self):
        return len(self.electrostatic_point_charges) > 0

    def is_magnetostatic(self):
        return len(self.magnetostatic_point_charges) > 0 or len(self.current_point_charges) > 0 
     
    def __add__(self, other):
        return self.__class__(
            self.electrostatic_point_charges.__add__(other.electrostatic_point_charges),
            self.magnetostatic_point_charges.__add__(other.magnetostatic_point_charges),
            self.current_point_charges.__add__(other.current_point_charges))
     
    def __sub__(self, other):
        return self.__class__(
            self.electrostatic_point_charges.__sub__(other.electrostatic_point_charges),
            self.magnetostatic_point_charges.__sub__(other.magnetostatic_point_charges),
            self.current_point_charges.__sub__(other.current_point_charges))
    
    def __radd__(self, other):
        return self.__class__(
            self.electrostatic_point_charges.__radd__(other.electrostatic_point_charges),
            self.magnetostatic_point_charges.__radd__(other.magnetostatic_point_charges),
            self.current_point_charges.__radd__(other.current_point_charges))
    
    def __mul__(self, other):
        if not _is_numeric(other):
            return NotImplemented
         
        return self.__class__(
            self.electrostatic_point_charges.__mul__(other),
            self.magnetostatic_point_charges.__mul__(other),
            self.current_point_charges.__mul__(other))
    
    def __neg__(self):
        return self.__class__(
            self.electrostatic_point_charges.__neg__(),
            self.magnetostatic_point_charges.__neg__(),
            self.current_point_charges.__neg__())
     
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
        return sum(self.area_of_element(i) for i in indices) 

    @abstractmethod
    def area_of_element(self, i: int) -> float:
        ...
    
    def charge_on_element(self, i):
        return self.area_of_element(i) * self.electrostatic_point_charges.charges[i]
    
    def charge_on_elements(self, indices):
        """Compute the sum of the charges present on the elements with the given indices. To
        get the total charge of a physical group use `names['name']` for indices where `names` 
        is returned by `traceon.excitation.Excitation.get_electrostatic_active_elements()`.

        Parameters
        ----------
        indices: (N,) array of int
            indices of the elements contributing to the charge sum. 
         
        Returns
        -------
        The sum of the charge. See the note about units on the front page."""
        return sum(self.charge_on_element(i) for i in indices)
    
    def __str__(self):
        name = self.__class__.__name__
        return f'<Traceon {name}\n' \
            f'\tNumber of electrostatic points: {len(self.electrostatic_point_charges)}\n' \
            f'\tNumber of magnetizable points: {len(self.magnetostatic_point_charges)}\n' \
            f'\tNumber of current rings: {len(self.current_point_charges)}>'
    
    @abstractmethod
    def current_field_at_point(self, point_):
        ...


class FieldRadialBEM(FieldBEM):
    """A radially symmetric electrostatic field. The field is a result of the surface charges as computed by the
    `solve_direct` function. See the comments in `FieldBEM`."""
    
    def __init__(self, electrostatic_point_charges=None, magnetostatic_point_charges=None, current_point_charges=None):
        if electrostatic_point_charges is None:
            electrostatic_point_charges = EffectivePointCharges.empty_2d()
        if magnetostatic_point_charges is None:
            magnetostatic_point_charges = EffectivePointCharges.empty_2d()
        if current_point_charges is None:
            current_point_charges = EffectivePointCharges.empty_3d()
         
        self.symmetry = E.Symmetry.RADIAL
        super().__init__(electrostatic_point_charges, magnetostatic_point_charges, current_point_charges)
         
    def current_field_at_point(self, point_):
        point = np.array(point_, dtype=np.double)
        assert point.shape == (3,), "Please supply a three dimensional point"
            
        currents = self.current_point_charges.charges
        jacobians = self.current_point_charges.jacobians
        positions = self.current_point_charges.positions
        return backend.current_field_radial(point, currents, jacobians, positions)
     
    def electrostatic_field_at_point(self, point_):
        """
        Compute the electric field, \\( \\vec{E} = -\\nabla \\phi \\)
        
        Parameters
        ----------
        point: (3,) array of float64
            Position at which to compute the field.
        
        Returns
        -------
        (3,) array of float64, containing the field strengths (units of V/m)
        """
        point = np.array(point_)
        assert point.shape == (3,), "Please supply a three dimensional point"
          
        charges = self.electrostatic_point_charges.charges
        jacobians = self.electrostatic_point_charges.jacobians
        positions = self.electrostatic_point_charges.positions
        return backend.field_radial(point, charges, jacobians, positions)
     
    def electrostatic_potential_at_point(self, point_):
        """
        Compute the electrostatic potential.
        
        Parameters
        ----------
        point: (3,) array of float64
            Position at which to compute the field.
        
        Returns
        -------
        Potential as a float value (in units of V).
        """
        point = np.array(point_)
        assert point.shape == (3,), "Please supply a three dimensional point"
        charges = self.electrostatic_point_charges.charges
        jacobians = self.electrostatic_point_charges.jacobians
        positions = self.electrostatic_point_charges.positions
        return backend.potential_radial(point, charges, jacobians, positions)
    
    def magnetostatic_field_at_point(self, point_):
        """
        Compute the magnetic field \\( \\vec{H} \\)
        
        Parameters
        ----------
        point: (3,) array of float64
            Position at which to compute the field.
             
        Returns
        -------
        (3,) np.ndarray of float64 containing the field strength (in units of A/m) in the x, y and z directions.
        """
        point = np.array(point_)
        assert point.shape == (3,), "Please supply a three dimensional point"
        current_field = self.current_field_at_point(point)
        
        charges = self.magnetostatic_point_charges.charges
        jacobians = self.magnetostatic_point_charges.jacobians
        positions = self.magnetostatic_point_charges.positions
        
        mag_field = backend.field_radial(point, charges, jacobians, positions)

        return current_field + mag_field

    def magnetostatic_potential_at_point(self, point_):
        """
        Compute the magnetostatic scalar potential (satisfying \\(\\vec{H} = -\\nabla \\phi \\))
        
        Parameters
        ----------
        point: (3,) array of float64
            Position at which to compute the field.
        
        Returns
        -------
        Potential as a float value (in units of A).
        """
        point = np.array(point_)
        assert point.shape == (3,), "Please supply a three dimensional point"
        charges = self.magnetostatic_point_charges.charges
        jacobians = self.magnetostatic_point_charges.jacobians
        positions = self.magnetostatic_point_charges.positions
        return backend.potential_radial(point, charges, jacobians, positions)
    
    def current_potential_axial(self, z):
        assert isinstance(z, float)
        currents = self.current_point_charges.charges
        jacobians = self.current_point_charges.jacobians
        positions = self.current_point_charges.positions
        return backend.current_potential_axial(z, currents, jacobians, positions)
     
    def get_electrostatic_axial_potential_derivatives(self, z):
        """
        Compute the derivatives of the electrostatic potential a points on the optical axis (z-axis). 
         
        Parameters
        ----------
        z : (N,) np.ndarray of float64
            Positions on the optical axis at which to compute the derivatives.

        Returns
        ------- 
        Numpy array of shape (N, 9) containing the derivatives. At index i one finds the i-th derivative (so
        at position 0 the potential itself is returned). The highest derivative returned is a 
        constant currently set to 9."""
        charges = self.electrostatic_point_charges.charges
        jacobians = self.electrostatic_point_charges.jacobians
        positions = self.electrostatic_point_charges.positions
        return backend.axial_derivatives_radial(z, charges, jacobians, positions)
    
    def get_magnetostatic_axial_potential_derivatives(self, z):
        """
        Compute the derivatives of the magnetostatic potential at points on the optical axis (z-axis). 
         
        Parameters
        ----------
        z : (N,) np.ndarray of float64
            Positions on the optical axis at which to compute the derivatives.

        Returns
        ------- 
        Numpy array of shape (N, 9) containing the derivatives. At index i one finds the i-th derivative (so
        at position 0 the potential itself is returned). The highest derivative returned is a 
        constant currently set to 9."""
        charges = self.magnetostatic_point_charges.charges
        jacobians = self.magnetostatic_point_charges.jacobians
        positions = self.magnetostatic_point_charges.positions
         
        derivs_magnetic = backend.axial_derivatives_radial(z, charges, jacobians, positions)
        derivs_current = self.get_current_axial_potential_derivatives(z)
        return derivs_magnetic + derivs_current
     
    def get_current_axial_potential_derivatives(self, z):
        """
        Compute the derivatives of the current magnetostatic scalar potential at points on the optical axis.
         
        Parameters
        ----------
        z : (N,) np.ndarray of float64
            Positions on the optical axis at which to compute the derivatives.
         
        Returns
        ------- 
        Numpy array of shape (N, 9) containing the derivatives. At index i one finds the i-th derivative (so
        at position 0 the potential itself is returned). The highest derivative returned is a 
        constant currently set to 9."""

        currents = self.current_point_charges.charges
        jacobians = self.current_point_charges.jacobians
        positions = self.current_point_charges.positions
        return backend.current_axial_derivatives_radial(z, currents, jacobians, positions)
      
    def area_of_element(self, i):
        jacobians = self.electrostatic_point_charges.jacobians
        positions = self.electrostatic_point_charges.positions
        return 2*np.pi*np.sum(jacobians[i] * positions[i, :, 0])
    
    def get_tracer(self, bounds):
        return T.Tracer(self, bounds)
    
    def get_low_level_trace_function(self):
        args = backend.FieldEvaluationArgsRadial(self.electrostatic_point_charges, self.magnetostatic_point_charges, self.current_point_charges, self.field_bounds)
        return backend.field_fun(("field_radial_traceable", backend.backend_lib)), args
        
    
FACTOR_AXIAL_DERIV_SAMPLING_2D = 0.2

class FieldAxial(Field, ABC):
    """An electrostatic field resulting from a radial series expansion around the optical axis. You should
    not initialize this class yourself, but it is used as a base class for the fields returned by the `axial_derivative_interpolation` methods. 
    This base class overloads the +,*,- operators so it is very easy to take a superposition of different fields."""
    
    def __init__(self, z, electrostatic_coeffs=None, magnetostatic_coeffs=None):
        N = len(z)
        assert z.shape == (N,)
        assert electrostatic_coeffs is None or len(electrostatic_coeffs)== N-1
        assert magnetostatic_coeffs is None or len(magnetostatic_coeffs) == N-1
        assert electrostatic_coeffs is not None or magnetostatic_coeffs is not None
        
        assert z[0] < z[-1], "z values in axial interpolation should be ascending"
         
        self.z = z
        self.electrostatic_coeffs = electrostatic_coeffs if electrostatic_coeffs is not None else np.zeros_like(magnetostatic_coeffs)
        self.magnetostatic_coeffs = magnetostatic_coeffs if magnetostatic_coeffs is not None else np.zeros_like(electrostatic_coeffs)
        
        self.has_electrostatic = np.any(self.electrostatic_coeffs != 0.)
        self.has_magnetostatic = np.any(self.magnetostatic_coeffs != 0.)
     
    def is_electrostatic(self):
        return self.has_electrostatic

    def is_magnetostatic(self):
        return self.has_magnetostatic
     
    def __str__(self):
        name = self.__class__.__name__
        return f'<Traceon {name}, zmin={self.z[0]} mm, zmax={self.z[-1]} mm,\n\tNumber of samples on optical axis: {len(self.z)}>'
     
    def __add__(self, other):
        if isinstance(other, FieldAxial):
            assert np.array_equal(self.z, other.z), "Cannot add FieldAxial if optical axis sampling is different."
            assert self.electrostatic_coeffs.shape == other.electrostatic_coeffs.shape, "Cannot add FieldAxial if shape of axial coefficients is unequal."
            assert self.magnetostatic_coeffs.shape == other.magnetostatic_coeffs.shape, "Cannot add FieldAxial if shape of axial coefficients is unequal."
            return self.__class__(self.z, self.electrostatic_coeffs+other.electrostatic_coeffs, self.magnetostatic_coeffs + other.magnetostatic_coeffs)
         
        return NotImplemented
    
    def __sub__(self, other):
        return self.__add__(-other)
    
    def __radd__(self, other):
        return self.__add__(other)
     
    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.z, other*self.electrostatic_coeffs, other*self.magnetostatic_coeffs)
         
        return NotImplemented
    
    def __neg__(self):
        return -1*self
    
    def __rmul__(self, other):
        return self.__mul__(other)

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


class FieldRadialAxial(FieldAxial):
    def __init__(self, field, zmin, zmax, N=None):
        """
        Produces a field which uses an axial interpolation to very quickly compute the field around the z-axis.
        Note that the approximation degrades as the point at which the field is computed is further from the z-axis.

        Parameters
        -----------------------
        field: `traceon.field.FieldRadialBEM`
            Field for which to compute the axial interpolation
        zmin : float
            Location on the optical axis where to start sampling the radial expansion coefficients.
        zmax : float
            Location on the optical axis where to stop sampling the radial expansion coefficients. Any field
            evaluation outside [zmin, zmax] will return a zero field strength.
        N: int, optional
            Number of samples to take on the optical axis, if N=None the amount of samples
            is determined by taking into account the number of elements in the mesh.
        """
        assert isinstance(field, FieldRadialBEM)

        z, electrostatic_coeffs, magnetostatic_coeffs = FieldRadialAxial._get_interpolation_coefficients(field, zmin, zmax, N=N)
        
        super().__init__(z, electrostatic_coeffs, magnetostatic_coeffs)
        
        assert self.electrostatic_coeffs.shape == (len(z)-1, backend.DERIV_2D_MAX, 6)
        assert self.magnetostatic_coeffs.shape == (len(z)-1, backend.DERIV_2D_MAX, 6)
    
    @staticmethod
    def _get_interpolation_coefficients(field: FieldRadialBEM, zmin, zmax, N=None):
        assert zmax > zmin, "zmax should be bigger than zmin"

        N_charges = max(len(field.electrostatic_point_charges.charges), len(field.magnetostatic_point_charges.charges))
        N = N if N is not None else int(FACTOR_AXIAL_DERIV_SAMPLING_2D*N_charges)
        z = np.linspace(zmin, zmax, N)
        
        st = time.time()
        elec_derivs = np.concatenate(util.split_collect(field.get_electrostatic_axial_potential_derivatives, z), axis=0)
        elec_coeffs = _quintic_spline_coefficients(z, elec_derivs.T)
        
        mag_derivs = np.concatenate(util.split_collect(field.get_magnetostatic_axial_potential_derivatives, z), axis=0)
        mag_coeffs = _quintic_spline_coefficients(z, mag_derivs.T)
        
        logging.log_info(f'Computing derivative interpolation took {(time.time()-st)*1000:.2f} ms ({len(z)} items)')

        return z, elec_coeffs, mag_coeffs
     
    def electrostatic_field_at_point(self, point_):
        """
        Compute the electric field, \\( \\vec{E} = -\\nabla \\phi \\)
        
        Parameters
        ----------
        point: (2,) array of float64
            Position at which to compute the field.
             
        Returns
        -------
        Numpy array containing the field strengths (in units of V/mm) in the r and z directions.
        """
        point = np.array(point_)
        assert point.shape == (3,), "Please supply a three dimensional point"
        return backend.field_radial_derivs(point, self.z, self.electrostatic_coeffs)
    
    def magnetostatic_field_at_point(self, point_):
        """
        Compute the magnetic field \\( \\vec{H} \\)
        
        Parameters
        ----------
        point: (2,) array of float64
            Position at which to compute the field.
             
        Returns
        -------
        (2,) np.ndarray of float64 containing the field strength (in units of A/m) in the x, y and z directions.
        """
        point = np.array(point_)
        assert point.shape == (3,), "Please supply a three dimensional point"
        return backend.field_radial_derivs(point, self.z, self.magnetostatic_coeffs)
     
    def electrostatic_potential_at_point(self, point_):
        """
        Compute the electrostatic potential (close to the axis).

        Parameters
        ----------
        point: (2,) array of float64
            Position at which to compute the potential.
        
        Returns
        -------
        Potential as a float value (in units of V).
        """
        point = np.array(point_)
        assert point.shape == (3,), "Please supply a three dimensional point"
        return backend.potential_radial_derivs(point, self.z, self.electrostatic_coeffs)
    
    def magnetostatic_potential_at_point(self, point_):
        """
        Compute the magnetostatic scalar potential (satisfying \\(\\vec{H} = -\\nabla \\phi \\)) close to the axis
        
        Parameters
        ----------
        point: (2,) array of float64
            Position at which to compute the field.
        
        Returns
        -------
        Potential as a float value (in units of A).
        """
        point = np.array(point_)
        assert point.shape == (3,), "Please supply a three dimensional point"
        return backend.potential_radial_derivs(point, self.z, self.magnetostatic_coeffs)
    
    def get_tracer(self, bounds):
        return T.Tracer(self, bounds)
    
    def get_low_level_trace_function(self):
        args = backend.FieldDerivsArgs(self.z, self.electrostatic_coeffs, self.magnetostatic_coeffs)
        return backend.field_fun(("field_radial_derivs_traceable", backend.backend_lib)), args
 
    

     
