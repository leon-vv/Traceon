"""The solver module uses the Boundary Element Method (BEM) to compute the surface charge distribution of a given
geometry and excitation. Once the surface charge distribution is known, the field at any arbitrary position in space
can be calculated by integration over the charged boundary. However, doing a field evaluation in this manner is very slow
as for every field evaluation an iteration needs to be done over all elements in the mesh. Especially for particle tracing it
is crucial that the field evaluation can be done faster. To achieve this, interpolation techniques can be used, see `traceon.field.FieldRadialAxial`,
and `traceon_pro.field.Field3DAxial`.
"""

__pdoc__ = {}
__pdoc__['EffectivePointCharges'] = False
__pdoc__['ElectrostaticSolver'] = False
__pdoc__['ElectrostaticSolverRadial'] = False
__pdoc__['MagnetostaticSolver'] = False
__pdoc__['MagnetostaticSolverRadial'] = False
__pdoc__['SolverRadial'] = False
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
from scipy.constants import e, mu_0, m_e

from . import geometry as G
from . import excitation as E
from . import logging
from . import backend
from . import util
from . import tracing as T
from .field import *

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
        
        
        
        self.jac_buffer, self.pos_buffer = self.get_jacobians_and_positions(self.vertices)

    @abstractmethod
    def get_jacobians_and_positions(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...
     
    @abstractmethod
    def get_active_elements(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        ...

    @abstractmethod
    def get_normal_vectors(self) -> np.ndarray:
        ...
    
    def get_number_of_matrix_elements(self):
        return len(self.vertices)
        
    def is_higher_order(self):
        return self.excitation.mesh.is_higher_order()
        
    def is_3d(self):
        return self.excitation.mesh.is_3d()
    
    def is_2d(self):
        return self.excitation.mesh.is_2d()

    def get_flux_indices(self):
        """Get the indices of the vertices that are of type DIELECTRIC or MAGNETIZABLE.
        For these indices we don't compute the potential but the flux through the element (the inner 
        product of the field with the normal of the vertex. The method is implemented in the derived classes."""
        ...
        N = self.get_number_of_matrix_elements()
        return np.arange(N)[ (self.excitation_types == int(E.ExcitationType.DIELECTRIC)) | (self.excitation_types == int(E.ExcitationType.MAGNETIZABLE)) ]
    
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
    def get_preexisting_field(self, point: np.ndarray) -> np.ndarray:
        """Get a field that exists even if all the charges are zero. This field
        is currently always a result of currents, but can in the future be extended
        to for example support permanent magnets."""
        ...
     
    def get_right_hand_side(self) -> np.ndarray:
        st = time.time()
        N = self.get_number_of_matrix_elements()
        F = np.zeros((N,))

        normals = self.get_normal_vectors()

        assert self.excitation_types.shape == (N,) and self.excitation_values.shape == (N,)

        # Loop over all excitation types/values
        for i, (type_, value) in enumerate(zip(self.excitation_types, self.excitation_values)):
            # Handle voltage-related excitations
            if type_ in [E.ExcitationType.VOLTAGE_FIXED, E.ExcitationType.VOLTAGE_FUN, E.ExcitationType.MAGNETOSTATIC_POT]:
                F[i] = value
            elif type_ == E.ExcitationType.DIELECTRIC:
                F[i] = 0
            elif type_ == E.ExcitationType.MAGNETIZABLE:
                center = self.get_center_of_element(i)
                field_at_center = self.get_preexisting_field(center)

                n = normals[i]
                # Convert 2D normal to 3D if needed
                if n.shape == (2,):
                    n = [n[0], 0.0, n[1]]

                field_dotted = np.dot(field_at_center, n)
                F[i] = -backend.flux_density_to_charge_factor(value) * field_dotted
        
        assert np.all(np.isfinite(F))
        
        logging.log_info(f"Computing right hand side of linear system took {(time.time() - st) * 1000:.0f} ms")

        return F
            
    @abstractmethod
    def charges_to_field(self, charges):
        ...

    @abstractmethod
    def get_matrix(self) -> np.ndarray:
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
        

class SolverRadial(Solver):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        N = len(self.vertices) 
        normals = np.zeros( (N, 2) if self.is_2d() else (N, 3) )
        
        for i, v in enumerate(self.vertices):
            if not self.is_higher_order():
                normals[i] = backend.normal_2d(v[0], v[1])
            else:
                normals[i] = backend.higher_order_normal_radial(0.0, v)
         
        self.normals = normals
    
    def get_normal_vectors(self):
        return self.normals
    
    def get_jacobians_and_positions(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return backend.fill_jacobian_buffer_radial(vertices)
    
    def get_matrix(self):
        # Sanity check: 2D must be higher-order
        assert self.is_2d() and self.is_higher_order(), "2D mesh needs to be higher-order (consider upgrading mesh)."
        
        N_matrix = self.get_number_of_matrix_elements()
        matrix = np.zeros((N_matrix, N_matrix))

        logging.log_info(
            f'Using matrix solver (radial). Number of elements: {N_matrix}, '
            f'size of matrix: {N_matrix} ({matrix.nbytes/1e6:.0f} MB), '
            f'symmetry: {self.excitation.symmetry}, '
            f'higher order: {self.excitation.mesh.is_higher_order()}')

        # For radial, use the radial fill function
        fill_fun = backend.fill_matrix_radial

        # Wrapper for filling matrix rows
        def fill_matrix_rows(rows):
            fill_fun(
                matrix,
                self.vertices,
                self.excitation_types,
                self.excitation_values,
                self.jac_buffer,
                self.pos_buffer,
                rows[0],
                rows[-1])

        # Fill the matrix
        st = time.time()
        util.split_collect(fill_matrix_rows, np.arange(N_matrix))

        # For radial meshes, we add self-potential/field contributions
        for i in range(N_matrix):
            type_ = self.excitation_types[i]
            val = self.excitation_values[i]

            if type_ in [E.ExcitationType.DIELECTRIC, E.ExcitationType.MAGNETIZABLE]:
                # -1 follows from matrix equation
                matrix[i, i] = (backend.self_field_dot_normal_radial(self.vertices[i], val) - 1)
            else:
                matrix[i, i] = backend.self_potential_radial(self.vertices[i])
        
        logging.log_info(f'Time for building radial matrix: {(time.time()-st)*1000:.0f} ms')
        
        assert np.all(np.isfinite(matrix)), "Matrix contains non-finite values."

        return matrix


class ElectrostaticSolverRadial(SolverRadial):
    def get_preexisting_field(self, point):
        np.zeros(3)
    
    def get_active_elements(self):
        return self.excitation.get_electrostatic_active_elements()
    
    def charges_to_field(self, charges):
        return FieldRadialBEM(electrostatic_point_charges=charges)


class MagnetostaticSolverRadial(SolverRadial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preexisting_field = self.get_current_field() + self.get_permanent_magnet_field()

    def get_active_elements(self):
        return self.excitation.get_magnetostatic_active_elements()
    
    def get_preexisting_field(self, point):
        return self.preexisting_field.magnetostatic_field_at_point(point)
     
    def get_permanent_magnet_field(self) -> FieldBEM:
        charges: list[np.ndarray] = []
        jacobians = []
        positions = []
        
        mesh = self.excitation.mesh
        
        if not len(mesh.lines) or not self.excitation.has_permanent_magnet():
            return FieldRadialBEM(magnetostatic_point_charges=EffectivePointCharges.empty_2d())
        
        all_vertices = mesh.points[mesh.lines]
        jac, pos = backend.fill_jacobian_buffer_radial(all_vertices)
        normals = np.array([backend.higher_order_normal_radial(0.0, v) for v in all_vertices])
        
        for name, v in self.excitation.excitation_types.items():
            if not v[0] == E.ExcitationType.PERMANENT_MAGNET or not name in mesh.physical_to_lines:
                continue
            
            indices = mesh.physical_to_lines[name]
            
            if not len(indices):
                continue

            # Magnetic charge is dot product of normal vector and magnetization vector
            n = normals[indices]
            assert n.shape == (len(n), 2)
            vector = v[1]
            dot_product = n[:, 0]*vector[0] + n[:, 1]*vector[2] # Normal currently has only (r,z) element
            
            charges.extend(dot_product)
            jacobians.extend(jac[indices])
            positions.extend(pos[indices])
        
        if not len(charges):
            return FieldRadialBEM(magnetostatic_point_charges=EffectivePointCharges.empty_2d())
        
        return FieldRadialBEM(magnetostatic_point_charges=EffectivePointCharges(np.array(charges), np.array(jacobians), np.array(positions)))
     
    def get_current_field(self) -> FieldBEM:
        currents: list[np.ndarray] = []
        jacobians = []
        positions = []
        
        mesh = self.excitation.mesh
        
        if not len(mesh.triangles) or not self.excitation.has_current():
            return FieldRadialBEM(current_point_charges=EffectivePointCharges.empty_3d())
         
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
            return FieldRadialBEM(current_point_charges=EffectivePointCharges.empty_3d())
        
        return FieldRadialBEM(current_point_charges=EffectivePointCharges(np.array(currents), np.array(jacobians), np.array(positions)))
    
    def charges_to_field(self, charges):
        return FieldRadialBEM(
            magnetostatic_point_charges=self.preexisting_field.magnetostatic_point_charges + charges,
            current_point_charges=self.preexisting_field.current_point_charges)

     
def _excitation_to_higher_order(excitation):
    logging.log_info('Upgrading mesh to higher to be compatible with matrix solver')
    # Upgrade mesh, such that matrix solver will support it
    excitation = copy.copy(excitation)
    mesh = copy.copy(excitation.mesh)
    excitation.mesh = mesh._to_higher_order_mesh()
    return excitation

def solve_direct_superposition(excitation):
    """
    When using superposition multiple fields are computed at once. Each field corresponds with a unity excitation (1V)
    of an electrode that was assigned a non-zero fixed voltage value. This is useful when a geometry needs
    to be analyzed for many different voltage settings. In this case taking a linear superposition of the returned fields
    allows to select a different voltage 'setting' without inducing any computational cost. There is no computational cost
    involved in using `superposition=True` since a direct solver is used which easily allows for multiple right hand sides (the
    matrix does not have to be inverted multiple times). However, voltage functions are invalid in the superposition process (position dependent voltages).

    Parameters
    ---------------------
    excitation: `traceon.excitation.Excitation`
        The excitation that produces the resulting field.
    
    Returns
    ---------------------------
    Dictionary from str to `traceon.field.Field`. Each key is the name of an electrode on which a voltage (or current) was applied, the corresponding values are the fields.
    """
    if excitation.mesh.is_2d() and not excitation.mesh.is_higher_order():
        excitation = _excitation_to_higher_order(excitation)
    
    # Speedup: invert matrix only once, when using superposition
    electrostatic_excitations, magnetostatic_excitations = excitation._split_for_superposition()
    
    # Solve for elec fields
    elec_names = electrostatic_excitations.keys()
    right_hand_sides = np.array([ElectrostaticSolverRadial(electrostatic_excitations[n]).get_right_hand_side() for n in elec_names])
    solutions = ElectrostaticSolverRadial(excitation).solve_matrix(right_hand_sides)
    elec_dict = {n:s for n, s in zip(elec_names, solutions)}
    
    # Solve for mag fields 
    mag_names = magnetostatic_excitations.keys()
    right_hand_sides = np.array([MagnetostaticSolverRadial(magnetostatic_excitations[n]).get_right_hand_side() for n in mag_names])
    solutions = MagnetostaticSolverRadial(excitation).solve_matrix(right_hand_sides)
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
    `FieldRadialBEM`
    """
    if excitation.mesh.is_2d() and not excitation.mesh.is_higher_order():
        excitation = _excitation_to_higher_order(excitation)
    
    mag, elec = excitation.is_magnetostatic(), excitation.is_electrostatic()

    assert mag or elec, "Solving for an empty excitation"

    if mag and elec:
        elec_field = ElectrostaticSolverRadial(excitation).solve_matrix()[0]
        mag_field = MagnetostaticSolverRadial(excitation).solve_matrix()[0]
        return elec_field + mag_field # type: ignore
    elif elec and not mag:
        return ElectrostaticSolverRadial(excitation).solve_matrix()[0]
    elif mag and not elec:
        return MagnetostaticSolverRadial(excitation).solve_matrix()[0]




