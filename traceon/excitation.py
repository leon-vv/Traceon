"""The excitation module allows to specify the excitation (or element types) of the different physical groups (electrodes)
created with the `traceon.geometry` module. 

The possible excitations are as follows:

- Voltage (either fixed or as a function of position)
- Dielectric, with arbitrary electric permittivity
- Current coil (radial symmetric geometry)
- Current lines (3D geometry)
- Magnetostatic scalar potential
- Magnetizable material, with arbitrary magnetic permeability
- Permanent magnet, with uniform magnetization

Once the excitation is specified, it can be passed to `traceon.solver.solve_direct` to compute the resulting field.
"""
from enum import IntEnum

import numpy as np
from scipy.constants import mu_0

from .backend import N_QUAD_2D
from .logging import log_error
from . import excitation as E

class Symmetry(IntEnum):
    """Symmetry to be used for solver. Used when deciding which formulas to use in the Boundary Element Method. The currently
    supported symmetries are radial symmetry (also called cylindrical symmetry) and general 3D geometries.
    """
    RADIAL = 0
    THREE_D = 2
    
    def __str__(self):
        if self == Symmetry.RADIAL:
            return 'radial'
        elif self == Symmetry.THREE_D:
            return '3d' 
    
    def is_2d(self):
        return self == Symmetry.RADIAL
        
    def is_3d(self):
        return self == Symmetry.THREE_D

class ExcitationType(IntEnum):
    """Possible excitation that can be applied to elements of the geometry. See the methods of `Excitation` for documentation."""
    VOLTAGE_FIXED = 1
    VOLTAGE_FUN = 2
    DIELECTRIC = 3
     
    CURRENT = 4
    MAGNETOSTATIC_POT = 5
    MAGNETIZABLE = 6
    PERMANENT_MAGNET = 7
     
    def is_electrostatic(self):
        return self in [ExcitationType.VOLTAGE_FIXED,
                        ExcitationType.VOLTAGE_FUN,
                        ExcitationType.DIELECTRIC]

    def is_magnetostatic(self):
        return self in [ExcitationType.MAGNETOSTATIC_POT,
                        ExcitationType.MAGNETIZABLE,
                        ExcitationType.CURRENT,
                        ExcitationType.PERMANENT_MAGNET]
     
    def __str__(self):
        if self == ExcitationType.VOLTAGE_FIXED:
            return 'voltage fixed'
        elif self == ExcitationType.VOLTAGE_FUN:
            return 'voltage function'
        elif self == ExcitationType.DIELECTRIC:
            return 'dielectric'
        elif self == ExcitationType.CURRENT:
            return 'current'
        elif self == ExcitationType.MAGNETOSTATIC_POT:
            return 'magnetostatic potential'
        elif self == ExcitationType.MAGNETIZABLE:
            return 'magnetizable'
        elif self == ExcitationType.PERMANENT_MAGNET:
            return 'permanent magnet'
         
        raise RuntimeError('ExcitationType not understood in __str__ method')
     

class Excitation:
    """ """
     
    def __init__(self, mesh, symmetry):
        self.mesh = mesh
        self.electrodes = mesh.get_electrodes()
        self.excitation_types = {}
        self.symmetry = symmetry
         
        if symmetry == Symmetry.RADIAL:
            assert self.mesh.points.shape[1] == 2 or np.all(self.mesh.points[:, 1] == 0.), \
                "When symmetry is RADIAL, the geometry should lie in the XZ plane"
    
    def __str__(self):
        return f'<Traceon Excitation,\n\t' \
            + '\n\t'.join([f'{n}={v} ({t})' for n, (t, v) in self.excitation_types.items()]) \
            + '>'

    def _ensure_electrode_is_lines(self, excitation_type, name):
        assert name in self.electrodes, f"Electrode '{name}' is not present in the mesh"
        assert name in self.mesh.physical_to_lines, f"Adding {excitation_type} excitation in {self.symmetry} symmetry is only supported if electrode '{name}' consists of lines"
    
    def _ensure_electrode_is_triangles(self, excitation_type, name):
        assert name in self.electrodes, f"Electrode '{name}' is not present in the mesh"
        assert name in self.mesh.physical_to_triangles, f"Adding {excitation_type} excitation in {self.symmetry} symmetry is only supported if electrode '{name}' consists of triangles"
     
    def add_voltage(self, **kwargs):
        """
        Apply a fixed voltage to the geometries assigned the given name.
        
        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary are the geometry names, while the values are the voltages in units of Volt. For example,
            calling the function as `add_voltage(lens=50)` assigns a 50V value to the geometry elements part of the 'lens' physical group.
            Alternatively, the value can be a function, which takes x, y, z coordinates as argument and returns the voltage at that position.
            Note that in 2D symmetries (such as radial symmetry) the z value for this function will always be zero.
        
        """
        for name, voltage in kwargs.items():
             
            if self.symmetry == E.Symmetry.RADIAL:
                self._ensure_electrode_is_lines('voltage', name)
            elif self.symmetry == E.Symmetry.THREE_D:
                self._ensure_electrode_is_triangles('voltage', name)
            
            if isinstance(voltage, int) or isinstance(voltage, float):
                self.excitation_types[name] = (ExcitationType.VOLTAGE_FIXED, voltage)
            elif callable(voltage):
                self.excitation_types[name] = (ExcitationType.VOLTAGE_FUN, voltage)
            else:
                raise NotImplementedError('Unrecognized voltage value')

    def add_current(self, **kwargs):
        """
        Apply a fixed total current to the geometries assigned the given name. Note that a coil is assumed,
        which implies that the current density is constant as a function of (r, z). In a solid piece of conducting material the current density would
        be higher at small r (as the 'loop' around the axis is shorter and therefore the resistance is lower).
        
        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary are the geometry names, while the values are the currents in units of Ampere. For example,
            calling the function as `add_current(coild=10)` assigns a 10A value to the geometry elements part of the 'coil' physical group.
        """
        if self.symmetry == Symmetry.RADIAL:
            for name, current in kwargs.items():
                self._ensure_electrode_is_triangles("current", name)
                self.excitation_types[name] = (ExcitationType.CURRENT, current)
        elif self.symmetry == Symmetry.THREE_D:
            for name, current in kwargs.items():
                self._ensure_electrode_is_lines("current", name)
                self.excitation_types[name] = (ExcitationType.CURRENT, current)
        else:
            raise ValueError('Symmetry should be one of RADIAL or THREE_D')

    def has_permanent_magnet(self):
        """Check whether the excitation contains a permanent magnet."""
        return any([t == ExcitationType.PERMANENT_MAGNET for t, _ in self.excitation_types.values()])
    
    def has_current(self):
        """Check whether a current is applied in this excitation."""
        return any([t == ExcitationType.CURRENT for t, _ in self.excitation_types.values()])
    
    def is_electrostatic(self):
        """Check whether the excitation contains electrostatic fields."""
        return any([t in [ExcitationType.VOLTAGE_FIXED, ExcitationType.VOLTAGE_FUN] for t, _ in self.excitation_types.values()])
     
    def is_magnetostatic(self):
        """Check whether the excitation contains magnetostatic fields."""
        return any([t in [ExcitationType.MAGNETOSTATIC_POT, ExcitationType.PERMANENT_MAGNET, ExcitationType.CURRENT] for t, _ in self.excitation_types.values()])
     
    def add_magnetostatic_potential(self, **kwargs):
        """
        Apply a fixed magnetostatic potential to the geometries assigned the given name.
        
        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary are the geometry names, while the values are the voltages in units of Ampere. For example,
            calling the function as `add_magnetostatic_potential(lens=50)` assigns a 50A value to the geometry elements part of the 'lens' physical group.
        """
        for name, pot in kwargs.items():
            if self.symmetry == E.Symmetry.RADIAL:
                self._ensure_electrode_is_lines('magnetostatic potential', name)
            elif self.symmetry == E.Symmetry.THREE_D:
                self._ensure_electrode_is_triangles('magnetostatic potential', name)
             
            self.excitation_types[name] = (ExcitationType.MAGNETOSTATIC_POT, pot)

    def add_magnetizable(self, **kwargs):
        """
        Assign a relative magnetic permeability to the geometries assigned the given name.
        
        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary are the geometry names, while the values are the relative dielectric constants. For example,
            calling the function as `add_dielectric(spacer=2)` assign the relative dielectric constant of 2 to the `spacer` physical group.
         
        """

        for name, permeability in kwargs.items():
            if self.symmetry == E.Symmetry.RADIAL:
                self._ensure_electrode_is_lines('magnetizable', name)
            elif self.symmetry == E.Symmetry.THREE_D:
                self._ensure_electrode_is_triangles('magnetizable', name)

            self.excitation_types[name] = (ExcitationType.MAGNETIZABLE, permeability)
    
    def add_permanent_magnet(self, **kwargs):
        """
        Assign a magnetization vector to a permanent magnet. The magnetization is supplied as the residual flux density vectors, with unit Tesla.
        
        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary are the geometry names, while the values are the residual flux density vectors (Numpy shape (3,)).
        """
        for name, vector in kwargs.items():
            vector = np.array(vector, dtype=np.float64) / mu_0 # Note that we convert from Tesla to A/m, since the rest of the code works with H fields (which has unit A/m)
            
            if self.symmetry == E.Symmetry.RADIAL:
                self._ensure_electrode_is_lines('permanent magnet', name)
                assert vector.shape == (3,) and vector[1] == 0.0 and vector[0] == 0.0, \
                    "Please supply the magnetization vector in radial symmetry as the vector [0, 0, B], with B" +\
                    " the residual flux density (unit Tesla). Note that a magnetization vector along r (for example [B, 0, 0]) " +\
                    " would lead to a non-uniform magnetization in radial symmetry, and is currently not supported. "

            elif self.symmetry == E.Symmetry.THREE_D:
                self._ensure_electrode_is_triangles('permanent magnet', name)
                assert vector.shape == (3,), "The magnetization vector must be a 3D vector."

            self.excitation_types[name] = (ExcitationType.PERMANENT_MAGNET, vector)
     
    def add_dielectric(self, **kwargs):
        """
        Assign a dielectric constant to the geometries assigned the given name.
        
        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary are the geometry names, while the values are the relative dielectric constants. For example,
            calling the function as `add_dielectric(spacer=2)` assign the relative dielectric constant of 2 to the `spacer` physical group.
         
        """
        for name, permittivity in kwargs.items():
            if self.symmetry == E.Symmetry.RADIAL:
                self._ensure_electrode_is_lines('dielectric', name)
            elif self.symmetry == E.Symmetry.THREE_D:
                self._ensure_electrode_is_triangles('dielectric', name)

            self.excitation_types[name] = (ExcitationType.DIELECTRIC, permittivity)

    def add_electrostatic_boundary(self, *args, ensure_inward_normals=True):
        """
        Specify geometry elements as electrostatic boundary elements. At the boundary we require E·n = 0 at every point on the boundary. This
        is equivalent to stating that the directional derivative of the electrostatic potential through the boundary is zero. Placing boundaries between
        the spaces of electrodes usually helps convergence tremendously. Note that a boundary is equivalent to a dielectric with a dielectric
        constant of zero. This is how a boundary is actually implemented internally.
        
        Parameters
        ----------
        *args: list of str
            The geometry names that should be considered a boundary.
        """
        if ensure_inward_normals:
            for electrode in args:
                self.mesh.ensure_inward_normals(electrode)
        
        for name in args:
            if self.symmetry == E.Symmetry.RADIAL:
                self._ensure_electrode_is_lines('electrostatic boundary', name)
            elif self.symmetry == E.Symmetry.THREE_D:
                self._ensure_electrode_is_triangles('electrostatic boundary', name)

        self.add_dielectric(**{a:0 for a in args})
    
    def add_magnetostatic_boundary(self, *args, ensure_inward_normals=True):
        """
        Specify geometry elements as magnetostatic boundary elements. At the boundary we require H·n = 0 at every point on the boundary. This
        is equivalent to stating that the directional derivative of the magnetostatic potential through the boundary is zero. Placing boundaries between
        the spaces of electrodes usually helps convergence tremendously. Note that a boundary is equivalent to a magnetic material with a magnetic 
        permeability of zero. This is how a boundary is actually implemented internally.
        
        Parameters
        ----------
        *args: list of str
            The geometry names that should be considered a boundary.
        """
        if ensure_inward_normals:
            for electrode in args:
                self.mesh.ensure_inward_normals(electrode)
         
        for name in args:
            if self.symmetry == E.Symmetry.RADIAL:
                self._ensure_electrode_is_lines('magnetostatic boundary', name)
            elif self.symmetry == E.Symmetry.THREE_D:
                self._ensure_electrode_is_triangles('magnetostatic boundary', name)
         
        self.add_magnetizable(**{a:0 for a in args})
    
    def _is_excitation_type_part_of_superposition(self, type_: ExcitationType) -> bool:
        # When computing a superposition, should we return a field for the given excitation type with
        # the given value? We should only return a field if the field is not trivially zero.
        # For example, an excitation with only a boundary element will never produce a field.
        # There are only a few cases that would produce a field:
        return type_ in [ExcitationType.VOLTAGE_FIXED, ExcitationType.VOLTAGE_FUN, ExcitationType.CURRENT, ExcitationType.PERMANENT_MAGNET]
     
    def _split_for_superposition(self):
        types = self.excitation_types.items()
        part_of_superposition = [(n, t.is_electrostatic()) for n, (t, v) in types if self._is_excitation_type_part_of_superposition(t)]
        
        electrostatic_excitations = {}
        magnetostatic_excitations = {}
         
        for (name, is_electrostatic) in part_of_superposition:
             
            new_types_dict = {}
             
            for n, (t, v) in types:
                if n == name or not self._is_excitation_type_part_of_superposition(t):
                    new_types_dict[n] = (t, v)
                elif t == ExcitationType.VOLTAGE_FUN: 
                    new_types_dict[n] = (t, lambda _: 0.0) # Already gets its own field, don't include in this one
                else: 
                    new_types_dict[n] = (t, np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0) # Already gets its own field, don't include in this one
            
            exc = Excitation(self.mesh, self.symmetry)
            exc.excitation_types = new_types_dict

            if is_electrostatic:
                electrostatic_excitations[name] = exc
            else:
                magnetostatic_excitations[name] = exc

        assert len(electrostatic_excitations) + len(magnetostatic_excitations) == len(part_of_superposition)
        return electrostatic_excitations, magnetostatic_excitations
    
    def _get_active_elements(self, type_):
        assert type_ in ['electrostatic', 'magnetostatic']
        
        if self.symmetry == Symmetry.RADIAL:
            elements = self.mesh.lines
            physicals = self.mesh.physical_to_lines
        else:
            elements = self.mesh.triangles
            physicals = self.mesh.physical_to_triangles

        def type_check(excitation_type):
            if type_ == 'electrostatic':
                return excitation_type.is_electrostatic()
            else:
                return excitation_type in [ExcitationType.MAGNETIZABLE, ExcitationType.MAGNETOSTATIC_POT]
        
        inactive = np.full(len(elements), True)
        for name, value in self.excitation_types.items():
            if type_check(value[0]):
                inactive[ physicals[name] ] = False
         
        map_index = np.arange(len(elements)) - np.cumsum(inactive)
        names = {n:map_index[i] for n, i in physicals.items() \
                    if n in self.excitation_types and type_check(self.excitation_types[n][0])}
         
        return self.mesh.points[ elements[~inactive] ], names
     
    def get_electrostatic_active_elements(self):
        """Get elements in the mesh that have an electrostatic excitation
        applied to them. 
         
        Returns
        --------
        A tuple of two elements: (points, names). points is a Numpy array of shape (N, 4, 3) in the case of 2D and (N, 3, 3) in the case of 3D. \
        This array contains the vertices of the line elements or the triangles. \
        Multiple points per line elements are used in the case of 2D since higher order BEM is employed, in which the true position on the line \
        element is given by a polynomial interpolation of the points. \
        names is a dictionary, the keys being the names of the physical groups mentioned by this excitation, \
        while the values are Numpy arrays of indices that can be used to index the points array.
        """
        return self._get_active_elements('electrostatic')
    
    def get_magnetostatic_active_elements(self):
        """Get elements in the mesh that have an magnetostatic excitation
        applied to them. This does not include current excitation, as these are not part of the matrix.
    
        Returns
        --------
        A tuple of two elements: (points, names). points is a Numpy array of shape (N, 4, 3) in the case of 2D and (N, 3, 3) in the case of 3D. \
        This array contains the vertices of the line elements or the triangles. \
        Multiple points per line elements are used in the case of 2D since higher order BEM is employed, in which the true position on the line \
        element is given by a polynomial interpolation of the points. \
        names is a dictionary, the keys being the names of the physical groups mentioned by this excitation, \
        while the values are Numpy arrays of indices that can be used to index the points array.
        """

        return self._get_active_elements('magnetostatic')
