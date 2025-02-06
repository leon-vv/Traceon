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
from __future__ import annotations
import time
from abc import ABC, abstractmethod
import copy
from itertools import product

import numpy as np
from scipy.interpolate import CubicSpline, BPoly, PPoly

from . import tracing as T
from . import excitation as E
from . import util
from . import logging
from . import backend
from .mesher import GeometricObject

from .typing import *

__pdoc__ = {}
__pdoc__['EffectivePointCharges'] = False
__pdoc__['Field.copy'] = False
__pdoc__['Field.get_low_level_trace_function'] = False
__pdoc__['FieldRadialBEM.get_low_level_trace_function'] = False
__pdoc__['FieldRadialAxial.get_low_level_trace_function'] = False

def _is_numeric(x):
    if isinstance(x, int) or isinstance(x, float) or isinstance(x, np.generic):
        return True


class EffectivePointCharges:
    def __init__(self, 
                 charges: ArrayFloat1D, 
                 jacobians: ArrayFloat2D, 
                 positions: ArrayFloat3D, 
                 directions: ArrayFloat2D | None = None) -> None:
        
        self.charges = np.array(charges, dtype=np.float64)
        self.jacobians = np.array(jacobians, dtype=np.float64)
        self.positions = np.array(positions, dtype=np.float64)
        self.directions = np.array(directions, dtype=np.float64) if directions is not None else None # Current elements will have a direction
        
        N = len(self.charges)
        N_QUAD = self.jacobians.shape[1]
        assert self.charges.shape == (N,) and self.jacobians.shape == (N, N_QUAD)
        assert self.positions.shape == (N, N_QUAD, 3) or self.positions.shape == (N, N_QUAD, 2)
        assert self.directions is None or self.directions.shape == (len(self.charges), N_QUAD, 3)
    
    @staticmethod 
    def empty_2d() -> EffectivePointCharges:
        N_QUAD_2D = backend.N_QUAD_2D
        return EffectivePointCharges(np.empty((0,)), np.empty((0, N_QUAD_2D)), np.empty((0,N_QUAD_2D,2)))

    @staticmethod 
    def empty_3d() -> EffectivePointCharges:
        N_TRIANGLE_QUAD = backend.N_TRIANGLE_QUAD
        return EffectivePointCharges(np.empty((0,)), np.empty((0, N_TRIANGLE_QUAD)), np.empty((0, N_TRIANGLE_QUAD, 3)))
    
    @staticmethod 
    def empty_line_3d() -> EffectivePointCharges:
        N_QUAD_2D = backend.N_QUAD_2D
        return EffectivePointCharges(np.empty((0,)), np.empty((0, N_QUAD_2D)), np.empty((0, N_QUAD_2D, 3)), np.empty((0, N_QUAD_2D, 3)))

    def is_2d(self) -> bool:
        return self.jacobians.shape[1] == backend.N_QUAD_2D
    
    def is_3d(self) -> bool:
        return self.jacobians.shape[1] == backend.N_TRIANGLE_QUAD
     
    def _matches_geometry(self, other: EffectivePointCharges) -> bool:
        return (self.positions.shape == other.positions.shape and np.allclose(self.positions, other.positions)
                and self.jacobians.shape == other.jacobians.shape and np.allclose(self.jacobians, other.jacobians))

    def __len__(self) -> int:
        return len(self.charges)
    
    def __add__(self, other: EffectivePointCharges) -> EffectivePointCharges:
        if not isinstance(other, EffectivePointCharges) or self.is_2d() != other.is_2d():
            return NotImplemented
        
        if self._matches_geometry(other):
            return EffectivePointCharges((self.charges + other.charges).astype(np.float64), self.jacobians, self.positions)
        else:
            return EffectivePointCharges(
                np.concatenate([self.charges, other.charges]),
                np.concatenate([self.jacobians, other.jacobians]),
                np.concatenate([self.positions, other.positions]))

    def __radd__(self, other: EffectivePointCharges) -> EffectivePointCharges:
        return self.__add__(other)
     
    def __mul__(self, other: float) -> EffectivePointCharges:
        if _is_numeric(other):
            return EffectivePointCharges(other*self.charges, self.jacobians, self.positions)
        
        return NotImplemented
    
    def __neg__(self) -> EffectivePointCharges:
        return -1*self
    
    def __sub__(self, other: EffectivePointCharges) -> EffectivePointCharges:
        return self.__add__(-other)
     
    def __rmul__(self, other: float) -> EffectivePointCharges:
        return self.__mul__(other)

    def __str__(self) -> str:
        dim = '2D' if self.is_2d() else '3D'
        return f'<EffectivePointCharges {dim}\n' \
               f'\tNumber of charges: {len(self.charges)}\n' \
               f'\tJacobian shape:  {self.jacobians.shape}\n' \
               f'\tPositions shape: {self.positions.shape}>'


class Field(GeometricObject, ABC):
    def __init__(self) -> None:
        self._origin = np.array([0,0,0], dtype=np.float64)
        self._basis = np.eye(3, dtype=np.float64)
        self._update_inverse_transformation_matrix()

        self.field_bounds: Bounds3D | None = None

    def get_origin(self) -> Point3D:
        """
        Get the origin of the field in the global coordinate system. This is the position
        that the origin (0, 0, 0) was transformed to by using methods from `traceon.mesher.GeometricObject`.

        Returns
        -----------------------------
        numpy.ndarray
            Float array of shape (3,)
        """
        return self._origin.copy()
    
    def get_basis(self) -> ArrayFloat2D:
        return self._basis.copy()

    def _update_inverse_transformation_matrix(self) -> None:
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = self._basis
        transformation_matrix[:3, 3] = self._origin

        assert np.linalg.det(transformation_matrix) != 0, ("Transformations of field have resulted in a two-dimensional coordinate system. "
                                                           "Please only use affine transformations.")
        self._inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    def copy(self) -> Self:
        return copy.copy(self)
    
    def map_points(self, fun: Callable[[PointLike3D], Point3D]) -> Self:
        field_copy = self.copy()
        
        field_copy._origin = fun(self._origin).astype(np.float64)
        assert field_copy._origin.shape == (3,), "Transformation of field did not map origin to a 3D point"
        
        field_copy._basis = np.array([fun(b + self._origin) - field_copy._origin for b in self._basis])
        assert field_copy._basis.shape == (3,3), "Transformation of field did not map unit vectors to a 3D vector"
        
        field_copy._update_inverse_transformation_matrix()
        return field_copy

    def map_points_to_local(self, point: PointLike3D) -> Point3D:
        """Converts a point from the global coordinate system to the local coordinate system of the field. 
    
        Parameters
        ---------------------
        point: (3,) np.ndarray of float64
            The coordinates of the point in the global coordinate system.

        Returns
        ---------------------
        (3,) np.ndarray of float64
            The coordinates of the point in the local coordinate system."""
        # represent the point in homogenous coordinates so we can do the inverse 
        # affine transformation with a single matrix multiplication.
        global_point_homogeneous = np.array([*point, 1.], dtype=np.float64)
        local_point_homogeneous = self._inverse_transformation_matrix @ global_point_homogeneous
        assert np.isclose(local_point_homogeneous[3], 1.)
        return local_point_homogeneous[:3]

    def set_bounds(self, bounds: BoundsLike3D, global_coordinates: bool = False) -> None:
        """Set the field bounds. Outside the field bounds the field always returns zero (i.e. no field). Note
        that even in 2D the field bounds needs to be specified for x,y and z axis. The trajectories in the presence
        of magnetostatic field are in general 3D even in radial symmetric geometries.
        
        Parameters
        -------------------
        bounds: (3, 2) np.ndarray of float64
            The min, max value of x, y, z respectively within the field is still computed.
        global_coordinates: bool
            If `True` the given bounds are in global coordinates and transformed to the fields local system internally.
        """
        bounds = np.array(bounds, dtype=np.float64)
        assert bounds.shape == (3,2)

        if global_coordinates:
            transformed_corners = np.array([self.map_points_to_local(corner) for corner in product(*bounds)])
            bounds = np.column_stack((transformed_corners.min(axis=0), transformed_corners.max(axis=0)))

        self.field_bounds = bounds
    
    def _within_field_bounds(self, point: PointLike3D) -> bool:
        return bool(self.field_bounds is None or np.all((self.field_bounds[:, 0] <= point) & (point <= self.field_bounds[:, 1])))

    def _matches_geometry(self, other: Field) -> bool:
        return False
    
    def field_at_point(self, point: PointLike3D) -> Vector3D:
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
            "use electrostatic_field_at_point or magnetostatic_field_at_point")
     
    def potential_at_point(self, point: Point3D) -> float:
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
            return self.magnetostatic_potential_at_point(point)
         
        raise RuntimeError("Cannot use potential_at_point when both electric and magnetic fields are present, " \
            "use electrostatic_potential_at_point or magnetostatic_potential_at_point")

    def electrostatic_field_at_point(self, point: PointLike3D) -> Vector3D:
        """
        Compute the electric field, \\( \\vec{E} = -\\nabla \\phi \\)
        
        Parameters
        ----------
        point: (3,) array of float64
            Position in global coordinate system at which to compute the field.
        
        Returns
        -------
        (3,) array of float64, containing the field strengths (units of V/m)
        """
        local_point = self.map_points_to_local(point)

        if self._within_field_bounds(local_point):
            return self._basis @ self.electrostatic_field_at_local_point(local_point)
        else:
             return np.array([0.,0.,0.])
        
    def magnetostatic_field_at_point(self, point: PointLike3D) -> Vector3D:
        """
        Compute the magnetic field \\( \\vec{H} \\)
        
        Parameters
        ----------
        point: (3,) array of float64
            Position in global coordinate system at which to compute the field.
             
        Returns
        -------
        (3,) np.ndarray of float64 containing the field strength (in units of A/m) in the x, y and z directions.
        """
        local_point = self.map_points_to_local(point)
        if self._within_field_bounds(local_point):
            return self._basis @ self.magnetostatic_field_at_local_point(local_point)
        else:
             return np.array([0.,0.,0.])
    
    def electrostatic_potential_at_point(self, point: PointLike3D) -> float:
        """
        Compute the electrostatic potential.
        
        Parameters
        ----------
        point: (3,) array of float64
            Position in global coordinate system at which to compute the field.
        
        Returns
        -------
        Potential as a float value (in units of V).
        """
        local_point = self.map_points_to_local(point)
        
        if self._within_field_bounds(local_point):
            return self.electrostatic_potential_at_local_point(local_point)
        else:
             return 0.

    def magnetostatic_potential_at_point(self, point: PointLike3D) -> float:
        """
        Compute the magnetostatic scalar potential (satisfying \\(\\vec{H} = -\\nabla \\phi \\))
        
        Parameters
        ----------
        point: (3,) array of float64
            Position in global coordinate system in local coordinate system at which to compute the field.
        
        Returns
        -------
        Potential as a float value (in units of A).
        """
        local_point = self.map_points_to_local(point)
        if self._within_field_bounds(local_point):
            return self.magnetostatic_potential_at_local_point(local_point)
        else:
             return 0.

    @abstractmethod
    def is_electrostatic(self) -> bool:
        ...
    
    @abstractmethod
    def is_magnetostatic(self) -> bool:
        ...
    
    @abstractmethod
    def electrostatic_field_at_local_point(self, point) -> Vector3D:
        ...

    @abstractmethod
    def magnetostatic_field_at_local_point(self, point) -> Vector3D:
        ...
    
    @abstractmethod
    def electrostatic_potential_at_local_point(self, point) -> float:
        ...
    
    @abstractmethod
    def magnetostatic_potential_at_local_point(self, point) -> float:
        ...
    
    # Following function can be implemented to get a speedup while tracing. 
    # Return a field function implemented in C and a ctypes argument needed. 
    # See the field_fun variable in backend/__init__.py.
    # Note that by default it gives back a Python function, which gives no speedup.
    def get_low_level_trace_function(self) -> tuple[Callable, Any] | tuple[Callable, Any, list[Any]]:
        fun = lambda pos, vel: (self.electrostatic_field_at_point(pos), self.magnetostatic_field_at_point(pos))
        return backend.wrap_field_fun(fun), None


class FieldSuperposition(Field):
    def __init__(self, fields: list[FieldBEM | FieldAxial]) -> None:
        assert all([isinstance(f, Field) for f in fields])
        self.fields = fields

    def map_points(self, fun: Callable[[PointLike3D], Point3D]) -> FieldSuperposition:
        return FieldSuperposition([f.map_points(fun) for f in self.fields])
    
    def field_at_point(self, point: PointLike3D) -> Vector3D:
        elec, mag = self.is_electrostatic(), self.is_magnetostatic()
        
        if elec and not mag:
            return self.electrostatic_field_at_point(point)
        elif not elec and mag:
            return self.magnetostatic_field_at_point(point)
        
        raise RuntimeError("Cannot use field_at_point when both electric and magnetic fields are present, " \
            "use electrostatic_field_at_point or magnetostatic_field_at_point")

    def potential_at_point(self, point: PointLike3D) -> float:
        elec, mag = self.is_electrostatic(), self.is_magnetostatic()

        if elec and not mag:
            return self.electrostatic_potential_at_point(point)
        elif not elec and mag:
            return self.magnetostatic_potential_at_point(point)
        
        raise RuntimeError("Cannot use potential_at_point when both electric and magnetic fields are present, " \
            "use electrostatic_potential_at_point or magnetostatic_potential_at_point")
        
    def electrostatic_field_at_point(self, point: PointLike3D) -> Vector3D:
        return np.sum([f.electrostatic_field_at_point(point) for f in self.fields], axis=0)

    def magnetostatic_field_at_point(self, point: PointLike3D) -> Vector3D:
        return np.sum([f.magnetostatic_field_at_point(point) for f in self.fields], axis=0)
    
    def current_field_at_point(self, point: PointLike3D) -> Vector3D:
        return np.sum([f.current_field_at_point(point) for f in self.fields if isinstance(f, FieldBEM)], axis=0) 

    def electrostatic_potential_at_point(self, point: PointLike3D) -> float:
        return sum([f.electrostatic_potential_at_point(point) for f in self.fields])

    def magnetostatic_potential_at_point(self, point: PointLike3D) -> float:
        return sum([f.magnetostatic_potential_at_point(point) for f in self.fields])
    
    def electrostatic_field_at_local_point(self, point: PointLike3D) -> Vector3D:
        return self.electrostatic_field_at_point(point)

    def magnetostatic_field_at_local_point(self, point: PointLike3D) -> Vector3D:
        return self.magnetostatic_field_at_point(point)
    
    def current_field_at_local_point(self, point: PointLike3D) -> Vector3D:
        return self.current_field_at_point(point)

    def electrostatic_potential_at_local_point(self, point: PointLike3D) -> float:
        return self.electrostatic_potential_at_point(point)

    def magnetostatic_potential_at_local_point(self, point: PointLike3D) -> float:
        return self.magnetostatic_potential_at_point(point)
    
    def is_electrostatic(self) -> bool:
        return any(f.is_electrostatic() for f in self.fields)
    
    def is_magnetostatic(self) -> bool:
        return any(f.is_magnetostatic() for f in self.fields)

    def get_tracer(self, bounds: BoundsLike3D) -> Tracer:
        return T.Tracer(self, bounds)

    def __add__(self, other: FieldBEM | FieldAxial | FieldSuperposition) -> FieldSuperposition:
        if isinstance(other, (FieldBEM, FieldAxial, FieldSuperposition)):
            other_fields = other.fields if isinstance(other, FieldSuperposition) else [other]
            fields_copy = self.fields.copy()
            for of in other_fields:
                for i, f in enumerate(self.fields):
                    if f._matches_geometry(of):
                        fields_copy[i] = cast(Union[FieldBEM, FieldAxial], f + of)
                        break
                else:
                    fields_copy.append(of)

            return FieldSuperposition(fields_copy)
        else:
            return NotImplemented
    
    def __iadd__(self, other: FieldBEM | FieldAxial | FieldSuperposition) -> FieldSuperposition:
        self.fields = (self + other).fields
        return self

    def __mul__(self, other: float) -> FieldSuperposition:
        if _is_numeric(other):
            return FieldSuperposition([f.__mul__(other) for f in self.fields])
        else:
            return NotImplemented
    
    def __rmul__(self, other: float) -> FieldSuperposition :
        return self.__mul__(other)
    
    def __getitem__(self, index: int | slice) -> FieldBEM | FieldAxial | FieldSuperposition:
        selection = np.array(self.fields, dtype=object).__getitem__(index)
        if isinstance(selection, np.ndarray):
            return FieldSuperposition(cast(list[FieldBEM | FieldAxial], selection.tolist()))
        else:
            return selection
    
    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Iterator[Field]:
        return iter(self.fields)
    
    def __str__(self) -> str:
        field_strs = '\n'.join(str(f) for f in self.fields)
        return f"<FieldSuperposition with fields:\n{field_strs}>"


class FieldBEM(Field, ABC):
    """An electrostatic field (resulting from surface charges) as computed from the Boundary Element Method. You should
    not initialize this class yourself, but it is used as a base class for the fields returned by the `solve_direct` function. 
    This base class overloads the +,*,- operators so it is very easy to take a superposition of different fields."""
    
    def __init__(self, 
                 electrostatic_point_charges: EffectivePointCharges, 
                 magnetostatic_point_charges: EffectivePointCharges, 
                 current_point_charges: EffectivePointCharges):
        
        super().__init__()
        
        self.electrostatic_point_charges = electrostatic_point_charges
        self.magnetostatic_point_charges = magnetostatic_point_charges
        self.current_point_charges = current_point_charges
        
    def is_electrostatic(self) -> bool:
        return len(self.electrostatic_point_charges) > 0

    def is_magnetostatic(self) -> bool:
        return len(self.magnetostatic_point_charges) > 0 or len(self.current_point_charges) > 0 
    
    def _matches_geometry(self, other: Field) -> bool:
        return (self.__class__ == other.__class__
            and np.allclose(self._origin, other._origin) 
            and np.allclose(self._basis, other._basis))


    def __add__(self, other: Field) -> FieldBEM | FieldSuperposition:
        if not isinstance(other, (FieldBEM, FieldAxial)):
            return NotImplemented
        
        if self._matches_geometry(other):
            other = cast(FieldBEM, other)
            field_copy = self.copy()
            field_copy.electrostatic_point_charges = self.electrostatic_point_charges + other.electrostatic_point_charges
            field_copy.magnetostatic_point_charges = self.magnetostatic_point_charges + other.magnetostatic_point_charges
            field_copy.current_point_charges = self.current_point_charges + other.current_point_charges
            return field_copy
        else:
            return FieldSuperposition([self, other])
        
    def __sub__(self, other: Field) -> FieldBEM | FieldSuperposition:
        if not isinstance(other, (FieldBEM, FieldAxial)):
            return NotImplemented
        
        return self.__add__(-other)

    def __radd__(self, other: Field) -> FieldBEM | FieldSuperposition:
        return self.__add__(other)
        
    def __mul__(self, other: float) -> FieldBEM:
        if _is_numeric(other):
           field_copy = self.copy()
           field_copy.electrostatic_point_charges = self.electrostatic_point_charges * other
           field_copy.magnetostatic_point_charges = self.magnetostatic_point_charges * other
           field_copy.current_point_charges = self.current_point_charges * other
           return field_copy
        else:
            return NotImplemented
    
    def __neg__(self) -> FieldBEM:
        return self.__class__(
            self.electrostatic_point_charges.__neg__(),
            self.magnetostatic_point_charges.__neg__(),
            self.current_point_charges.__neg__())
     
    def __rmul__(self, other: float) -> FieldBEM:
        return self.__mul__(other)
      
    def area_of_elements(self, indices: ArrayLikeInt1D):
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
    
    def charge_on_element(self, i: int) -> float:
        return self.area_of_element(i) * self.electrostatic_point_charges.charges[i]
    
    def charge_on_elements(self, indices: ArrayLikeInt1D) -> float:
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
    
    def __str__(self) -> str:
        name = self.__class__.__name__
        return f'<Traceon {name}\n' \
            f'\tNumber of electrostatic points: {len(self.electrostatic_point_charges)}\n' \
            f'\tNumber of magnetizable points: {len(self.magnetostatic_point_charges)}\n' \
            f'\tNumber of current rings: {len(self.current_point_charges)}>'
    
    def current_field_at_point(self, point: PointLike3D) -> Vector3D:
        local_point = self.map_points_to_local(point)
        if (self.field_bounds is None or np.all((self.field_bounds[:, 0] <= local_point) 
                                                & (local_point <= self.field_bounds[:, 1]))):
            return self.current_field_at_local_point(local_point)
        else:
             return np.zeros(3)

    @abstractmethod
    def current_field_at_local_point(self, point: PointLike3D) -> Point3D:
        ...

    
class FieldRadialBEM(FieldBEM):
    """A radially symmetric electrostatic field. The field is a result of the surface charges as computed by the
    `solve_direct` function. See the comments in `FieldBEM`."""
    
    def __init__(self, 
                 electrostatic_point_charges: EffectivePointCharges | None = None, 
                 magnetostatic_point_charges: EffectivePointCharges | None = None, 
                 current_point_charges: EffectivePointCharges | None = None) -> None:
        
        if electrostatic_point_charges is None:
            electrostatic_point_charges = EffectivePointCharges.empty_2d()
        if magnetostatic_point_charges is None:
            magnetostatic_point_charges = EffectivePointCharges.empty_2d()
        if current_point_charges is None:
            current_point_charges = EffectivePointCharges.empty_3d()
        
        assert all([isinstance(eff, EffectivePointCharges) for eff in [electrostatic_point_charges,
                                                                       magnetostatic_point_charges,
                                                                       current_point_charges]])
        self.symmetry = E.Symmetry.RADIAL
        super().__init__(electrostatic_point_charges, magnetostatic_point_charges, current_point_charges)
         
    def current_field_at_local_point(self, point: PointLike3D) -> Vector3D:
        point = np.array(point, dtype=np.float64)
        assert point.shape == (3,), "Please supply a three dimensional point"
            
        currents = self.current_point_charges.charges
        jacobians = self.current_point_charges.jacobians
        positions = self.current_point_charges.positions
        return backend.current_field_radial(point, currents, jacobians, positions)
     
    def electrostatic_field_at_local_point(self, point: PointLike3D) -> Vector3D:
        """
        Compute the electric field, \\( \\vec{E} = -\\nabla \\phi \\)
        
        Parameters
        ----------
        point: (3,) array of float64
            Position in local coordinate system at which to compute the field.
        
        Returns
        -------
        (3,) array of float64, containing the field strengths (units of V/m)
        """
        point = np.array(point, dtype=np.float64)
        assert point.shape == (3,), "Please supply a three dimensional point"
          
        charges = self.electrostatic_point_charges.charges
        jacobians = self.electrostatic_point_charges.jacobians
        positions = self.electrostatic_point_charges.positions
        return backend.field_radial(point, charges, jacobians, positions)
    
    def magnetostatic_field_at_local_point(self, point: PointLike3D) -> Vector3D:
        """
        Compute the magnetic field \\( \\vec{H} \\)
        
        Parameters
        ----------
        point: (3,) array of float64
            Position in local coordinate system at which to compute the field.
             
        Returns
        -------
        (3,) np.ndarray of float64 containing the field strength (in units of A/m) in the x, y and z directions.
        """
        point = np.array(point, dtype=np.float64)
        assert point.shape == (3,), "Please supply a three dimensional point"
        current_field = self.current_field_at_point(point)
        
        charges = self.magnetostatic_point_charges.charges
        jacobians = self.magnetostatic_point_charges.jacobians
        positions = self.magnetostatic_point_charges.positions
        
        mag_field = backend.field_radial(point, charges, jacobians, positions)

        return current_field + mag_field

    def electrostatic_potential_at_local_point(self, point: PointLike3D) -> float:
        """
        Compute the electrostatic potential.
        
        Parameters
        ----------
        point: (3,) array of float64
            Position in local coordinate system at which to compute the field.
        
        Returns
        -------
        Potential as a float value (in units of V).
        """
        point = np.array(point, dtype=np.float64)
        assert point.shape == (3,), "Please supply a three dimensional point"
        charges = self.electrostatic_point_charges.charges
        jacobians = self.electrostatic_point_charges.jacobians
        positions = self.electrostatic_point_charges.positions
        return backend.potential_radial(point, charges, jacobians, positions)
     
    def magnetostatic_potential_at_local_point(self, point: PointLike3D) -> float:
        """
        Compute the magnetostatic scalar potential (satisfying \\(\\vec{H} = -\\nabla \\phi \\))
        
        Parameters
        ----------
        point: (3,) array of float64
            Position in local coordinate system in local coordinate system at which to compute the field.
        
        Returns
        -------
        Potential as a float value (in units of A).
        """
        point = np.array(point, dtype=np.float64)
        assert point.shape == (3,), "Please supply a three dimensional point"
        charges = self.magnetostatic_point_charges.charges
        jacobians = self.magnetostatic_point_charges.jacobians
        positions = self.magnetostatic_point_charges.positions
        return backend.potential_radial(point, charges, jacobians, positions)
    
    def current_potential_axial(self, z: float) -> float:
        assert isinstance(z, float)
        currents = self.current_point_charges.charges
        jacobians = self.current_point_charges.jacobians
        positions = self.current_point_charges.positions
        return backend.current_potential_axial(z, currents, jacobians, positions)
     
    def get_electrostatic_axial_potential_derivatives(self, z: ArrayLikeFloat1D) -> ArrayFloat2D:
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
        z = np.array(z, dtype=np.float64)
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
     
    def get_current_axial_potential_derivatives(self, z: ArrayLikeFloat1D) -> ArrayFloat2D:
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
        z = np.array(z, dtype=np.float64)
        currents = self.current_point_charges.charges
        jacobians = self.current_point_charges.jacobians
        positions = self.current_point_charges.positions
        return backend.current_axial_derivatives_radial(z, currents, jacobians, positions)
      
    def area_of_element(self, i: int) -> float:
        jacobians = self.electrostatic_point_charges.jacobians
        positions = self.electrostatic_point_charges.positions
        return 2*np.pi*np.sum(jacobians[i] * positions[i, :, 0])
    
    def get_tracer(self, bounds: BoundsLike3D)-> Tracer:
        return T.Tracer(self, bounds)
    
    def get_low_level_trace_function(self) -> tuple[Callable, Any]:
        args = backend.FieldEvaluationArgsRadial(self.electrostatic_point_charges, self.magnetostatic_point_charges, self.current_point_charges, self.field_bounds)
        return backend.field_fun(("field_radial_traceable", backend.backend_lib)), args


FACTOR_AXIAL_DERIV_SAMPLING_2D = 0.2

class FieldAxial(Field, ABC):
    """An electrostatic field resulting from a radial series expansion around the optical axis. You should
    not initialize this class yourself, but it is used as a base class for the fields returned by the `axial_derivative_interpolation` methods. 
    This base class overloads the +,*,- operators so it is very easy to take a superposition of different fields."""
    
    def __init__(self, 
                 field: FieldBEM, 
                 z: ArrayFloat1D, 
                 electrostatic_coeffs: ArrayFloat3D | None = None, 
                 magnetostatic_coeffs: ArrayFloat3D | None = None):
        
        super().__init__()
        self.field = field
        self._origin = field._origin
        self._basis = field._basis
        self._update_inverse_transformation_matrix()

        N = len(z)
        assert z.shape == (N,)
        assert electrostatic_coeffs is None or len(electrostatic_coeffs)== N-1
        assert magnetostatic_coeffs is None or len(magnetostatic_coeffs) == N-1
        assert electrostatic_coeffs is not None or magnetostatic_coeffs is not None
        
        assert z[0] < z[-1], "z values in axial interpolation should be ascending"

        self.z = z
        self.electrostatic_coeffs = electrostatic_coeffs if electrostatic_coeffs is not None else np.zeros_like(magnetostatic_coeffs)
        self.magnetostatic_coeffs = magnetostatic_coeffs if magnetostatic_coeffs is not None else np.zeros_like(electrostatic_coeffs)
        
        self.has_electrostatic = bool(np.any(self.electrostatic_coeffs != 0.))
        self.has_magnetostatic = bool(np.any(self.magnetostatic_coeffs != 0.))
     
    def is_electrostatic(self) -> bool:
        return self.has_electrostatic

    def is_magnetostatic(self) -> bool:
        return self.has_magnetostatic
    
    def _matches_geometry(self, other: Field) -> bool:
        if (self.__class__ != other.__class__):
            return False
        else:
            other = cast(FieldAxial, other)
            return(np.allclose(self._origin, other._origin) 
                and np.allclose(self._basis, other._basis)
                and self.z.shape == other.z.shape and np.allclose(self.z, other.z))
    
    def __str__(self) -> str:
        name = self.__class__.__name__
        return f'<Traceon {name}, zmin={self.z[0]} mm, zmax={self.z[-1]} mm,\n\tNumber of samples on optical axis: {len(self.z)}>'
     
    def __add__(self, other: Field) -> FieldAxial | FieldSuperposition:
        if not isinstance(other, (FieldBEM, FieldAxial)):
            return NotImplemented
        
        if self._matches_geometry(other):
            other = cast(FieldAxial, other)
            field_copy = self.copy()
            field_copy.electrostatic_coeffs = self.electrostatic_coeffs + other.electrostatic_coeffs
            field_copy.magnetostatic_coeffs = self.magnetostatic_coeffs + other.magnetostatic_coeffs
            return field_copy
        else:
            return FieldSuperposition([self, other])

    def __sub__(self, other: Field) -> FieldAxial | FieldSuperposition:
        if not isinstance(other, (FieldBEM, FieldAxial)): 
            return NotImplemented
        
        return self.__add__(-other)

    def __radd__(self, other: Field) -> FieldAxial | FieldSuperposition:
        return self.__add__(other)
     
    def __mul__(self, other: float) -> FieldAxial:
        if _is_numeric(other):
            field_copy = self.copy()
            field_copy.electrostatic_coeffs = other * self.electrostatic_coeffs
            field_copy.magnetostatic_coeffs = other * self.electrostatic_coeffs
            return field_copy
        else:
            return NotImplemented
    
    def __neg__(self) -> FieldAxial:
        return -1*self
    
    def __rmul__(self, other: float) -> FieldAxial:
        return self.__mul__(other)

def _get_one_dimensional_high_order_ppoly(z: ArrayLikeFloat1D, 
                                          y: float , 
                                          dydz: float, 
                                          dydz2: float) -> PPoly:
    
    bpoly = BPoly.from_derivatives(z, np.array([y, dydz, dydz2]).T)
    return PPoly.from_bernstein_basis(bpoly)        

def _quintic_spline_coefficients(z: ArrayLikeFloat1D, derivs: ArrayLikeFloat1D) -> ArrayFloat3D:
    # k is degree of polynomial
    #assert derivs.shape == (z.size, backend.DERIV_2D_MAX)
    z = np.array(z, dtype=np.float64)
    c = np.zeros( (z.size-1, 9, 6) )
    
    dz = z[1] - z[0]
    assert np.all(np.isclose(np.diff(z), dz)) # Equally spaced
     
    for i, d in enumerate(cast(List[int], derivs)):
        high_order = i + 2 < len(derivs)
        
        if high_order:
            ppoly = _get_one_dimensional_high_order_ppoly(z, d, derivs[i+1], derivs[i+2])
            start_index = 0
        else:
            ppoly = CubicSpline(z, d) # type: ignore
            start_index = 2
        
        c[:, i, start_index:], x, k = ppoly.c.T, ppoly.x, ppoly.c.shape[0]-1
        assert np.all(x == z)
        assert (high_order and k == 5) or (not high_order and k == 3)
    
    return c


class FieldRadialAxial(FieldAxial):
    def __init__(self, field: FieldRadialBEM, zmin: float, zmax: float, N: int | None = None) -> None:
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
        
        super().__init__(field, z, electrostatic_coeffs, magnetostatic_coeffs)

        assert self.electrostatic_coeffs.shape == (len(z)-1, backend.DERIV_2D_MAX, 6)
        assert self.magnetostatic_coeffs.shape == (len(z)-1, backend.DERIV_2D_MAX, 6)
    
    @staticmethod
    def _get_interpolation_coefficients(field: FieldRadialBEM, 
                                        zmin: float, 
                                        zmax: float, 
                                        N: int | None = None) -> tuple[ArrayFloat1D, ArrayFloat3D, ArrayFloat3D]:
        
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
     
    def electrostatic_field_at_local_point(self, point: PointLike3D) -> Vector3D:
        """
        Compute the electric field, \\( \\vec{E} = -\\nabla \\phi \\)
        
        Parameters
        ----------
        point: (3,) array of float64
            Position in local coordinate system at which to compute the field.
             
        Returns
        -------
        Numpy array containing the field strengths (in units of V/mm) in the r and z directions.
        """
        point = np.array(point, dtype=np.float64)
        assert point.shape == (3,), "Please supply a three dimensional point"
        return backend.field_radial_derivs(point, self.z, self.electrostatic_coeffs)
    
    def magnetostatic_field_at_local_point(self, point: PointLike3D) -> Vector3D:
        """
        Compute the magnetic field \\( \\vec{H} \\)
        
        Parameters
        ----------
        point: (3,) array of float64
            Position in local coordinate system at which to compute the field.
             
        Returns
        -------
        (3,) np.ndarray of float64 containing the field strength (in units of A/m) in the x, y and z directions.
        """
        point = np.array(point, dtype=np.float64)
        assert point.shape == (3,), "Please supply a three dimensional point"
        return backend.field_radial_derivs(point, self.z, self.magnetostatic_coeffs)
     
    def electrostatic_potential_at_local_point(self, point: PointLike3D) -> float:
        """
        Compute the electrostatic potential (close to the axis).

        Parameters
        ----------
        point: (3,) array of float64
            Position at which to compute the potential.
        
        Returns
        -------
        Potential as a float value (in units of V).
        """
        point = np.array(point)
        assert point.shape == (3,), "Please supply a three dimensional point"
        return backend.potential_radial_derivs(point, self.z, self.electrostatic_coeffs)
    
    def magnetostatic_potential_at_local_point(self, point: PointLike3D) -> float:
        """
        Compute the magnetostatic scalar potential (satisfying \\(\\vec{H} = -\\nabla \\phi \\)) close to the axis
        
        Parameters
        ----------
        point: (3,) array of float64
            Position in local coordinate system at which to compute the potential.
        
        Returns
        -------
        Potential as a float value (in units of A).
        """
        point = np.array(point, dtype=np.float64)
        assert point.shape == (3,), "Please supply a three dimensional point"
        return backend.potential_radial_derivs(point, self.z, self.magnetostatic_coeffs)
    
    def get_tracer(self, bounds: BoundsLike3D) -> Tracer:
        return T.Tracer(self, bounds)
    
    def get_low_level_trace_function(self) -> tuple[Callable, Any]:
        args = backend.FieldDerivsArgs(self.z, self.electrostatic_coeffs, self.magnetostatic_coeffs)
        return backend.field_fun(("field_radial_derivs_traceable", backend.backend_lib)), args
 
    

