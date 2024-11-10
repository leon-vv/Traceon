import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import CubicSpline, BPoly, PPoly

from . import solver as S
from . import tracing as T
from . import util
from . import logging
from . import backend

FACTOR_AXIAL_DERIV_SAMPLING_2D = 0.2

class FieldAxial(S.Field, ABC):
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
    """ """
    def __init__(self, field, zmin, zmax, N=None):
        assert isinstance(field, S.FieldRadialBEM)

        z, electrostatic_coeffs, magnetostatic_coeffs = FieldRadialAxial._get_interpolation_coefficients(field, zmin, zmax, N=N)
        
        super().__init__(z, electrostatic_coeffs, magnetostatic_coeffs)
        
        assert self.electrostatic_coeffs.shape == (len(z)-1, backend.DERIV_2D_MAX, 6)
        assert self.magnetostatic_coeffs.shape == (len(z)-1, backend.DERIV_2D_MAX, 6)
    
    @staticmethod
    def _get_interpolation_coefficients(field: S.FieldRadialBEM, zmin, zmax, N=None):
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
        return T.TracerRadialAxial(self, bounds)
    

