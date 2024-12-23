"""The solver module uses the Boundary Element Method (BEM) to compute the surface charge distribution of a given
geometry and excitation. Once the surface charge distribution is known, the field at any arbitrary position in space
can be calculated by integration over the charged boundary. However, doing a field evaluation in this manner is very slow
as for every field evaluation an iteration needs to be done over all elements in the mesh. Especially for particle tracing it
is crucial that the field evaluation can be done faster. To achieve this, interpolation techniques can be used. 

The solver package offers interpolation in the form of _radial series expansions_ to drastically increase the speed of ray tracing. For
this consider the `axial_derivative_interpolation` methods documented below.

## Radial series expansion in cylindrical symmetry

Let \\( \\phi_0(z) \\) be the potential along the optical axis. We can express the potential around the optical axis as:

$$
\\phi = \\phi_0(z_0) - \\frac{r^2}{4} \\frac{\\partial \\phi_0^2}{\\partial z^2} + \\frac{r^4}{64} \\frac{\\partial^4 \\phi_0}{\\partial z^4} - \\frac{r^6}{2304} \\frac{\\partial \\phi_0^6}{\\partial z^6} + \\cdots
$$

Therefore, if we can efficiently compute the axial potential derivatives \\( \\frac{\\partial \\phi_0^n}{\\partial z^n} \\) we can compute the potential and therefore the fields around the optical axis.
For the derivatives of \\( \\phi_0(z) \\) closed form formulas exist in the case of radially symmetric geometries, see for example formula 13.16a in [1]. Traceon uses a recursive version of these formulas to
very efficiently compute the axial derivatives of the potential.

## Radial series expansion in 3D

In a general three dimensional geometry the potential will be dependent not only on the distance from the optical axis but also on the angle \\( \\theta \\) around the optical axis
at which the potential is sampled. It turns out (equation (35, 24) in [2]) the potential can be written as follows:

$$
\\phi = \\sum_{\\nu=0}^\\infty \\sum_{m=0}^\\infty r^{2\\nu + m} \\left( A^\\nu_m \\cos(m\\theta) + B^\\nu_m \\sin(m\\theta) \\right)
$$

The \\(A^\\nu_m\\) and \\(B^\\nu_m\\) coefficients can be expressed in _directional derivatives_ perpendicular to the optical axis, analogous to the radial symmetric case. The 
mathematics of calculating these coefficients quickly and accurately gets quite involved, but all details have been abstracted away from the user.

### References
[1] P. Hawkes, E. Kasper. Principles of Electron Optics. Volume one: Basic Geometrical Optics. 2018.

[2] W. Glaser. Grundlagen der Elektronenoptik. 1952.

"""

__pdoc__ = {}
__pdoc__['EffectivePointCharges'] = False
__pdoc__['ElectrostaticSolver'] = False
__pdoc__['MagnetostaticSolver'] = False
__pdoc__['Solver'] = False

import math as m
import time
from threading import Thread
import os.path as path
import copy
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
from scipy.special import legendre

from . import geometry as G
from . import excitation as E
from . import logging
from . import backend
from . import util
from . import tracing as T


class Solver(ABC):
    
    def __init__(self, excitation):
        self.excitation = excitation
        vertices, names = self.get_active_elements()
        
        self.excitation = excitation
        self.vertices = vertices
        self.names = names
         
        N = len(vertices)
        excitation_types = np.zeros(N, dtype=np.uint8)
        excitation_values = np.zeros(N)
         
        for n, indices in names.items():
            type_ = excitation.excitation_types[n][0]
            excitation_types[indices] = int(type_)
            
            if type_ != E.ExcitationType.VOLTAGE_FUN:
                excitation_values[indices] = excitation.excitation_types[n][1]
            else:
                function = excitation.excitation_types[n][1]
                excitation_values[indices] = [function(*self.get_center_of_element(i)) for i in indices]
          
        self.excitation_types = excitation_types
        self.excitation_values = excitation_values

        two_d = self.is_2d()
        higher_order = self.is_higher_order()

        assert not higher_order or two_d, "Higher order not supported in 3D"
        
        if two_d and higher_order:
            jac, pos = backend.fill_jacobian_buffer_radial(vertices)
        elif not two_d:
            jac, pos = backend.fill_jacobian_buffer_3d(vertices)
        else:
            raise ValueError('Input excitation is 2D but not higher order, this solver input is currently not supported. Consider upgrading mesh to higher order.')
        
        self.jac_buffer = jac
        self.pos_buffer = pos
     
    @abstractmethod
    def get_active_elements(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        ...
    
    def get_number_of_matrix_elements(self):
        return len(self.vertices)
        
    def is_higher_order(self):
        return self.excitation.mesh.is_higher_order()
        
    def is_3d(self):
        return self.excitation.mesh.is_3d()
    
    def is_2d(self):
        return self.excitation.mesh.is_2d()

    @abstractmethod
    def get_flux_indices(self):
        """Get the indices of the vertices that are of type DIELECTRIC or MAGNETIZABLE.
        For these indices we don't compute the potential but the flux through the element (the inner 
        product of the field with the normal of the vertex. The method is implemented in the derived classes."""
        ...
         
    def get_center_of_element(self, index):
        two_d = self.is_2d()
        higher_order = self.is_higher_order()
         
        if self.is_3d() or not self.is_higher_order():
            return np.mean(self.vertices[index], axis=0)
        else:
            v0, v1, v2, v3 = self.vertices[index]
            jac, pos = backend.position_and_jacobian_radial(0, v0, v2, v3, v1)
            return np.array([pos[0], 0.0, pos[1]])
     
    @abstractmethod
    def get_right_hand_side(self):
        ...
         
    def get_matrix(self):
        assert (self.is_3d() and not self.is_higher_order()) or \
            (self.is_2d() and self.is_higher_order()), "2D mesh needs to be higher order (consider upgrading mesh), 3D mesh needs to be simple (higher order not supported)."
         
        N_matrix = self.get_number_of_matrix_elements()
        matrix = np.zeros( (N_matrix, N_matrix) )
        logging.log_info(f'Using matrix solver, number of elements: {N_matrix}, size of matrix: {N_matrix} ({matrix.nbytes/1e6:.0f} MB), symmetry: {self.excitation.symmetry}, higher order: {self.excitation.mesh.is_higher_order()}')
         
        fill_fun = backend.fill_matrix_3d if self.is_3d() else backend.fill_matrix_radial
        
        def fill_matrix_rows(rows):
            fill_fun(matrix,
                self.vertices,
                self.excitation_types,
                self.excitation_values,
                self.jac_buffer, self.pos_buffer, rows[0], rows[-1])
        
        st = time.time()
        util.split_collect(fill_matrix_rows, np.arange(N_matrix))    
        logging.log_info(f'Time for building matrix: {(time.time()-st)*1000:.0f} ms')

        if not self.is_3d():
            # Technical detail: radial cannot compute their own self potential/field
            # need to fill it in here
            for i in range(N_matrix):
                type_ = self.excitation_types[i]
                val = self.excitation_values[i]
                
                if type_ == E.ExcitationType.DIELECTRIC or type_ == E.ExcitationType.MAGNETIZABLE:
                    # -1 follows from matrix equation
                    matrix[i, i] = backend.self_field_dot_normal_radial(self.vertices[i], self.excitation_values[i]) - 1
                else:
                    matrix[i, i] = backend.self_potential_radial(self.vertices[i])
        
        assert np.all(np.isfinite(matrix))
         
        return matrix
    
    @abstractmethod
    def charges_to_field(self, charges):
        ...
         
    def solve_matrix(self, right_hand_side=None):
        F = np.array([self.get_right_hand_side()]) if right_hand_side is None else right_hand_side
        
        N = self.get_number_of_matrix_elements()
         
        if N == 0:
            return [self.charges_to_field(EffectivePointCharges.empty_2d() if self.is_2d() else EffectivePointCharges.empty_3d()) \
                        for _ in range(len(F))]
        
        assert all([f.shape == (N,) for f in F])
        matrix = self.get_matrix()
         
        st = time.time()
        charges = np.linalg.solve(matrix, F.T).T
        logging.log_info(f'Time for solving matrix: {(time.time()-st)*1000:.0f} ms')
        assert np.all(np.isfinite(charges)) and charges.shape == F.shape
        
        result = [self.charges_to_field(EffectivePointCharges(c, self.jac_buffer, self.pos_buffer)) for c in charges]
         
        assert len(result) == len(F)
        return result
        
class ElectrostaticSolver(Solver):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_flux_indices(self):
        N = self.get_number_of_matrix_elements()
        return np.arange(N)[self.excitation_types == int(E.ExcitationType.DIELECTRIC)]

    def get_active_elements(self):
        return self.excitation.get_electrostatic_active_elements()
     
    def get_right_hand_side(self):
        N = self.get_number_of_matrix_elements()
        F = np.zeros( (N,) )
         
        assert self.excitation_types.shape ==(N,) and self.excitation_values.shape == (N,)
         
        # TODO: optimize in backend?
        for i, (type_, value) in enumerate(zip(self.excitation_types, self.excitation_values)):
            if type_ in [E.ExcitationType.VOLTAGE_FIXED, E.ExcitationType.VOLTAGE_FUN]:
                F[i] = value
            elif type_ == E.ExcitationType.DIELECTRIC:
                F[i] = 0
         
        assert np.all(np.isfinite(F))
        return F

    def charges_to_field(self, charges):
        if self.is_3d():
            return Field3D_BEM(electrostatic_point_charges=charges)
        else:
            return FieldRadialBEM(electrostatic_point_charges=charges)

class MagnetostaticSolver(Solver):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
         
        # Field produced by the current excitations on the coils
        self.current_charges = self.get_current_charges()
        self.current_field = FieldRadialBEM(current_point_charges=self.current_charges)
         
        # TODO: optimize in backend?
        N = len(self.vertices) 
        normals = np.zeros( (N, 2) if self.is_2d() else (N, 3) )
        
        for i, v in enumerate(self.vertices):
            if self.is_2d() and not self.is_higher_order():
                normals[i] = backend.normal_2d(v[0], v[1])
            elif self.is_2d() and self.is_higher_order():
                normals[i] = backend.higher_order_normal_radial(0.0, v[:, :2])
            elif self.is_3d() and not self.is_higher_order():
                normals[i] = backend.normal_3d(v)
        
        self.normals = normals
    
    def get_active_elements(self):
        return self.excitation.get_magnetostatic_active_elements()
    
    def get_flux_indices(self):
        N = self.get_number_of_matrix_elements()
        return np.arange(N)[self.excitation_types == int(E.ExcitationType.MAGNETIZABLE)]
     
    def get_current_charges(self):
        currents: list[np.ndarray] = []
        jacobians = []
        positions = []
        
        mesh = self.excitation.mesh
        
        if not len(mesh.triangles) or not self.excitation.has_current():
            return EffectivePointCharges.empty_3d()
         
        jac, pos = backend.fill_jacobian_buffer_3d(mesh.points[mesh.triangles])
        
        for n, v in self.excitation.excitation_types.items():
            if not v[0] == E.ExcitationType.CURRENT or not n in mesh.physical_to_triangles:
                continue
            
            indices = mesh.physical_to_triangles[n]
            
            if not len(indices):
                continue
             
            # Current supplied is total current, not a current density, therefore
            # divide by the area.
            area = np.sum(jac[indices])
            currents.extend(np.full(len(indices), v[1]/area))
            jacobians.extend(jac[indices])
            positions.extend(pos[indices])
        
        if not len(currents):
            return EffectivePointCharges.empty_3d()
        
        return EffectivePointCharges(np.array(currents), np.array(jacobians), np.array(positions))
     
    def get_right_hand_side(self):
        st = time.time()
        N = self.get_number_of_matrix_elements()
        F = np.zeros( (N,) )
         
        assert self.excitation_types.shape ==(N,) and self.excitation_values.shape == (N,)
         
        # TODO: optimize in backend?
        for i, (type_, value) in enumerate(zip(self.excitation_types, self.excitation_values)):
            if type_ == E.ExcitationType.MAGNETOSTATIC_POT:
                F[i] = value
            elif type_ == E.ExcitationType.MAGNETIZABLE:
                # Here we compute the inner product of the field generated by the current excitations
                # and the normal vector of the vertex.
                center = self.get_center_of_element(i)
                field_at_center = self.current_field.current_field_at_point(center)
                #flux_to_charge_factor = (value - 1)/np.pi
                field_dotted = field_at_center[0] * self.normals[i, 0] + field_at_center[2]*self.normals[i, 1]
                F[i] = -backend.flux_density_to_charge_factor(value) * field_dotted
         
        assert np.all(np.isfinite(F))
        logging.log_info(f'Computing right hand side of linear system took {(time.time()-st)*1000:.0f} ms')
        return F
     
    def charges_to_field(self, charges):
        if self.is_3d():
            return Field3D_BEM(magnetostatic_point_charges=charges)
        else:
            return FieldRadialBEM(magnetostatic_point_charges=charges, current_point_charges=self.current_charges)

class EffectivePointCharges:
    def __init__(self, charges, jacobians, positions):
        self.charges = np.array(charges, dtype=np.float64)
        self.jacobians = np.array(jacobians, dtype=np.float64)
        self.positions = np.array(positions, dtype=np.float64)
         
        N = len(self.charges)
        N_QUAD = self.jacobians.shape[1]
        assert self.charges.shape == (N,) and self.jacobians.shape == (N, N_QUAD)
        assert self.positions.shape == (N, N_QUAD, 3) or self.positions.shape == (N, N_QUAD, 2)
    
    @staticmethod 
    def empty_2d():
        N_QUAD_2D = backend.N_QUAD_2D
        return EffectivePointCharges(np.empty((0,)), np.empty((0, N_QUAD_2D)), np.empty((0,N_QUAD_2D,2)))

    @staticmethod 
    def empty_3d():
        N_TRIANGLE_QUAD = backend.N_TRIANGLE_QUAD
        return EffectivePointCharges(np.empty((0,)), np.empty((0, N_TRIANGLE_QUAD)), np.empty((0, N_TRIANGLE_QUAD, 3)))

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
    
    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return EffectivePointCharges(other*self.charges, self.jacobians, self.positions)
        
        return NotImplemented
    
    def __neg__(self):
        return -1*self
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        dim = '2D' if self.is_2d() else '3D'
        return f'<EffectivePointCharges {dim}\n' \
               f'\tNumber of charges: {len(self.charges)}\n' \
               f'\tJacobian shape:  {self.jacobians.shape}\n' \
               f'\tPositions shape: {self.positions.shape}>'

        
     
def _excitation_to_higher_order(excitation):
    logging.log_info('Upgrading mesh to higher to be compatible with matrix solver')
    # Upgrade mesh, such that matrix solver will support it
    excitation = copy.copy(excitation)
    mesh = copy.copy(excitation.mesh)
    excitation.mesh = mesh._to_higher_order_mesh()
    return excitation

def solve_direct_superposition(excitation):
    """
    superposition : bool
        When using superposition the function returns multiple fields. Each field corresponds with a unity excitation (1V)
        of a physical group that was previously assigned a non-zero fixed voltage value. This is useful when a geometry needs
        to be analyzed for many different voltage settings. In this case taking a linear superposition of the returned fields
        allows to select a different voltage 'setting' without inducing any computational cost. There is no computational cost
        involved in using `superposition=True` since a direct solver is used which easily allows for multiple right hand sides (the
        matrix does not have to be inverted multiple times). However, voltage functions are invalid in the superposition process (position dependent voltages).
    """
    if excitation.mesh.is_2d() and not excitation.mesh.is_higher_order():
        excitation = _excitation_to_higher_order(excitation)
    
    # Speedup: invert matrix only once, when using superposition
    excitations = excitation._split_for_superposition()
    
    # Solve for elec fields
    elec_names = [n for n, v in excitations.items() if v.is_electrostatic()]
    right_hand_sides = np.array([ElectrostaticSolver(excitations[n]).get_right_hand_side() for n in elec_names])
    solutions = ElectrostaticSolver(excitation).solve_matrix(right_hand_sides)
    elec_dict = {n:s for n, s in zip(elec_names, solutions)}
    
    # Solve for mag fields 
    mag_names = [n for n, v in excitations.items() if v.is_magnetostatic()]
    right_hand_sides = np.array([MagnetostaticSolver(excitations[n]).get_right_hand_side() for n in mag_names])
    solutions = MagnetostaticSolver(excitation).solve_matrix(right_hand_sides)
    mag_dict = {n:s for n, s in zip(mag_names, solutions)}
        
    return {**elec_dict, **mag_dict}

def solve_direct(excitation):
    """
    Solve for the charges on the surface of the geometry by using a direct method and taking
    into account the specified `excitation`. 

    Parameters
    ----------
    excitation : traceon.excitation.Excitation
        The excitation that produces the resulting field.
     
    Returns
    -------
    A `FieldRadialBEM` if the geometry (contained in the given `excitation`) is radially symmetric. If the geometry is a three
    dimensional geometry `Field3D_BEM` is returned. 
    """
    if excitation.mesh.is_2d() and not excitation.mesh.is_higher_order():
        excitation = _excitation_to_higher_order(excitation)
    
    mag, elec = excitation.is_magnetostatic(), excitation.is_electrostatic()

    assert mag or elec, "Solving for an empty excitation"
        
    if mag and elec:
        elec_field = ElectrostaticSolver(excitation).solve_matrix()[0]
        mag_field = MagnetostaticSolver(excitation).solve_matrix()[0]
        return elec_field + mag_field # type: ignore
    elif elec and not mag:
        return ElectrostaticSolver(excitation).solve_matrix()[0]
    elif mag and not elec:
        return MagnetostaticSolver(excitation).solve_matrix()[0]



class Field(ABC):
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
            return self.magnetostatic_potential_at_point(point)
         
        raise RuntimeError("Cannot use potential_at_point when both electric and magnetic fields are present, " \
            "use electrostatic_potential_at_point or magnetostatic_potential_at_point")

    @abstractmethod
    def is_electrostatic(self):
        ...
    
    @abstractmethod
    def is_magnetostatic(self):
        ...
    
    @abstractmethod
    def magnetostatic_potential_at_point(self, point):
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
        self.field_bounds = np.array(bounds)
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
        return self.__class__(
            self.electrostatic_point_charges.__mul__(other.electrostatic_point_charges),
            self.magnetostatic_point_charges.__mul__(other.magnetostatic_point_charges),
            self.current_point_charges.__mul__(other.current_point_charges))
    
    def __neg__(self, other):
        return self.__class__(
            self.electrostatic_point_charges.__neg__(other.electrostatic_point_charges),
            self.magnetostatic_point_charges.__neg__(other.magnetostatic_point_charges),
            self.current_point_charges.__neg__(other.current_point_charges))
     
    def __rmul__(self, other):
        return self.__class__(
            self.electrostatic_point_charges.__rmul__(other.electrostatic_point_charges),
            self.magnetostatic_point_charges.__rmul__(other.magnetostatic_point_charges),
            self.current_point_charges.__rmul__(other.current_point_charges))
     
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
        point = np.array(point_)
        assert point.shape == (3,), "Please supply a three dimensional point"
            
        currents = self.current_point_charges.charges
        jacobians = self.current_point_charges.jacobians
        positions = self.current_point_charges.positions
        return backend.current_field(point, currents, jacobians, positions)
     
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
        return T.TracerRadialBEM(self, bounds)
    
    
class Field3D_BEM(FieldBEM):
    """An electrostatic field resulting from a general 3D geometry. The field is a result of the surface charges as computed by the
    `solve_direct` function. See the comments in `FieldBEM`."""
     
    def __init__(self, electrostatic_point_charges=None, magnetostatic_point_charges=None):
        
        if electrostatic_point_charges is None:
            electrostatic_point_charges = EffectivePointCharges.empty_3d()
        if magnetostatic_point_charges is None:
            magnetostatic_point_charges = EffectivePointCharges.empty_3d()
         
        super().__init__(electrostatic_point_charges, magnetostatic_point_charges, EffectivePointCharges.empty_3d())
        
        self.symmetry = E.Symmetry.THREE_D

        for eff in [electrostatic_point_charges, magnetostatic_point_charges]:
            N = len(eff.charges)
            assert eff.charges.shape == (N,)
            assert eff.jacobians.shape == (N, backend.N_TRIANGLE_QUAD)
            assert eff.positions.shape == (N, backend.N_TRIANGLE_QUAD, 3)
     
    def electrostatic_field_at_point(self, point_):
        """
        Compute the electric field, \\( \\vec{E} = -\\nabla \\phi \\)
        
        Parameters
        ----------
        point: (3,) array of float64
            Position at which to compute the field.
             
        Returns
        -------
        (3,) array of float64 representing the electric field 
        """
        point = np.array(point_)
        assert point.shape == (3,), "Please supply a three dimensional point"
        charges = self.electrostatic_point_charges.charges
        jacobians = self.electrostatic_point_charges.jacobians
        positions = self.electrostatic_point_charges.positions
        return backend.field_3d(point, charges, jacobians, positions)
     
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
        return backend.potential_3d(point, charges, jacobians, positions)
     
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
        charges = self.magnetostatic_point_charges.charges
        jacobians = self.magnetostatic_point_charges.jacobians
        positions = self.magnetostatic_point_charges.positions
        return backend.field_3d(point, charges, jacobians, positions)
     
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
        return backend.potential_3d(point, charges, jacobians, positions)
     
    def area_of_element(self, i):
        jacobians = self.electrostatic_point_charges.jacobians
        return np.sum(jacobians[i])
    
    def get_tracer(self, bounds):
        return T.Tracer3D_BEM(self, bounds)
     

class FieldAxial(Field):
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

                
class FieldRadialAxial(FieldAxial):
    """ """
    def __init__(self, z, electrostatic_coeffs=None, magnetostatic_coeffs=None):
        super().__init__(z, electrostatic_coeffs, magnetostatic_coeffs)
        assert self.electrostatic_coeffs.shape == (len(z)-1, backend.DERIV_2D_MAX, 6)
        assert self.magnetostatic_coeffs.shape == (len(z)-1, backend.DERIV_2D_MAX, 6)
        self.symmetry = E.Symmetry.RADIAL
    
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
        return backend.field_radial_derivs(point, self.z, self.electrostatic_coeffs)
    
    def magnetostatic_field_at_point(self, point_):
        """
        Compute the magnetic field \\( \\vec{H} \\)
        
        Parameters
        ----------
        point: (3,) array of float64
            Position at which to compute the field.
             
        Returns
        -------
        (3,) array of float64, containing the field strengths (units of A/m)
        """
        point = np.array(point_)
        assert point.shape == (3,), "Please supply a three dimensional point"
        return backend.field_radial_derivs(point, self.z, self.magnetostatic_coeffs)
     
    def electrostatic_potential_at_point(self, point_):
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
        point = np.array(point_)
        assert point.shape == (3,), "Please supply a three dimensional point"
        return backend.potential_radial_derivs(point, self.z, self.electrostatic_coeffs)
    
    def magnetostatic_potential_at_point(self, point_):
        """
        Compute the magnetostatic scalar potential (satisfying \\(\\vec{H} = -\\nabla \\phi \\)) close to the axis
        
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
        return backend.potential_radial_derivs(point, self.z, self.magnetostatic_coeffs)
    
    def get_tracer(self, bounds):
        return T.TracerRadialAxial(self, bounds)


