"""The excitation module allows to specify the excitation (or element types) of the different physical groups (electrodes)
created with the `traceon.geometry` module. 

The possible excitations are as follows:

- Fixed voltage (electrode connect to a power supply)
- Voltage function (a generic Python function specifies the voltage as a function of position)
- Dielectric, with arbitrary electric permittivity
- Current coil, with fixed total amount of current (only in radial symmetry)
- Magnetostatic scalar potential
- Magnetizable material, with arbitrary magnetic permeability

Currently current excitations are not supported in 3D. But magnetostatic fields can still be computed using the magnetostatic scalar potential.

Once the excitation is specified, it can be passed to `traceon.solver.solve_bem` to compute the resulting field.
"""


from enum import IntEnum

import numpy as np

from .geometry import Symmetry
from .backend import N_QUAD_2D

class ExcitationType(IntEnum):
    """Possible excitation that can be applied to elements of the geometry. See the methods of `Excitation` for documentation."""
    VOLTAGE_FIXED = 1
    VOLTAGE_FUN = 2
    DIELECTRIC = 3
     
    CURRENT = 4
    MAGNETOSTATIC_POT = 5
    MAGNETIZABLE = 6
     
    def is_electrostatic(self):
        return self in [ExcitationType.VOLTAGE_FIXED,
                        ExcitationType.VOLTAGE_FUN,
                        ExcitationType.DIELECTRIC]

    def is_magnetostatic(self):
        return self in [ExcitationType.MAGNETOSTATIC_POT,
                        ExcitationType.MAGNETIZABLE,
                        ExcitationType.CURRENT]
     
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
         
        raise RuntimeError('ExcitationType not understood in __str__ method')
     

class Excitation:
    """ """
     
    def __init__(self, mesh):
        self.mesh = mesh
        self.electrodes = mesh.get_electrodes()
        self.excitation_types = {}
    
    def __str__(self):
        return f'<Traceon Excitation,\n\t' \
            + '\n\t'.join([f'{n}={v} ({t})' for n, (t, v) in self.excitation_types.items()]) \
            + '>'
     
    def add_voltage(self, **kwargs):
        """
        Apply a fixed voltage to the geometries assigned the given name (or physical group in GMSH terminology).
        
        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary are the geometry names, while the values are the voltages in units of Volt. For example,
            calling the function as `add_voltage(lens=50)` assigns a 50V value to the geometry elements part of the 'lens' physical group.
            Alternatively, the value can be a function, which takes x, y, z coordinates as argument and returns the voltage at that position.
            Note that in 2D symmetries (such as radial symmetry) the z value for this function will always be zero.
        
        """
        for name, voltage in kwargs.items():
            assert name in self.electrodes
            if isinstance(voltage, int) or isinstance(voltage, float):
                self.excitation_types[name] = (ExcitationType.VOLTAGE_FIXED, voltage)
            elif callable(voltage):
                self.excitation_types[name] = (ExcitationType.VOLTAGE_FUN, voltage)
            else:
                raise NotImplementedError('Unrecognized voltage value')

    def add_current(self, **kwargs):
        """
        Apply a fixed total current to the geometries assigned the given name (or physical group in GMSH terminology). Note that a coil is assumed,
        which implies that the current density is constant as a function of (r, z). In a solid piece of conducting material the current density would
        be higher at small r (as the 'loop' around the axis is shorter and therefore the resistance is lower).
        
        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary are the geometry names, while the values are the currents in units of Ampere. For example,
            calling the function as `add_current(coild=10)` assigns a 10A value to the geometry elements part of the 'coil' physical group.
        """

        assert self.mesh.symmetry == Symmetry.RADIAL, "Currently magnetostatics are only supported for radially symmetric meshes"
         
        for name, current in kwargs.items():
            assert name in self.mesh.physical_to_triangles.keys(), "Current can only be applied to a triangle electrode"
            self.excitation_types[name] = (ExcitationType.CURRENT, current)

    def has_current(self):
        """Check whether a current is applied in this excitation."""
        return any([t == ExcitationType.CURRENT for t, _ in self.excitation_types.values()])
    
    def is_electrostatic(self):
        """Check whether the excitation contains electrostatic fields."""
        return any([t in [ExcitationType.VOLTAGE_FIXED, ExcitationType.VOLTAGE_FUN] for t, _ in self.excitation_types.values()])
     
    def is_magnetostatic(self):
        """Check whether the excitation contains magnetostatic fields."""
        return any([t in [ExcitationType.MAGNETOSTATIC_POT, ExcitationType.CURRENT] for t, _ in self.excitation_types.values()])
     
    def add_magnetostatic_potential(self, **kwargs):
        """
        Apply a fixed magnetostatic potential to the geometries assigned the given name (or physical group in GMSH terminology).
        
        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary are the geometry names, while the values are the voltages in units of Ampere. For example,
            calling the function as `add_magnetostatic_potential(lens=50)` assigns a 50A value to the geometry elements part of the 'lens' physical group.
        """
        for name, pot in kwargs.items():
            assert name in self.electrodes
            self.excitation_types[name] = (ExcitationType.MAGNETOSTATIC_POT, pot)

    def add_magnetizable(self, **kwargs):
        """
        Assign a relative magnetic permeability to the geometries assigned the given name (or physical group in GMSH terminology).
        
        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary are the geometry names, while the values are the relative dielectric constants. For example,
            calling the function as `add_dielectric(spacer=2)` assign the relative dielectric constant of 2 to the `spacer` physical group.
         
        """

        for name, permeability in kwargs.items():
            assert name in self.electrodes
            self.excitation_types[name] = (ExcitationType.MAGNETIZABLE, permeability)
     
    def add_dielectric(self, **kwargs):
        """
        Assign a dielectric constant to the geometries assigned the given name (or physical group in GMSH terminology).
        
        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary are the geometry names, while the values are the relative dielectric constants. For example,
            calling the function as `add_dielectric(spacer=2)` assign the relative dielectric constant of 2 to the `spacer` physical group.
         
        """
        for name, permittivity in kwargs.items():
            assert name in self.electrodes
            self.excitation_types[name] = (ExcitationType.DIELECTRIC, permittivity)

    def add_electrostatic_boundary(self, *args):
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
        self.add_dielectric(**{a:0 for a in args})
    
    def add_magnetostatic_boundary(self, *args):
        """
        Specify geometry elements as magnetostatic boundary elements. At the boundary we require H·n = 0 at every point on the boundary. This
        is equivalent to stating that the directional derivative of the magnetostatic potential through the boundary is zero. Placing boundaries between
        the spaces of electrodes usually helps convergence tremendously. 
        
        Parameters
        ----------
        *args: list of str
            The geometry names that should be considered a boundary.
        """

        self.add_magnetizable(**{a:0 for a in args})
    
    def _split_for_superposition(self):
        
        # Names that have a fixed voltage excitation, not equal to 0.0
        types = self.excitation_types
        non_zero_fixed = [n for n, (t, v) in types.items() if t in [ExcitationType.VOLTAGE_FIXED,
                                                                    ExcitationType.CURRENT] and v != 0.0]
        
        excitations = []
         
        for name in non_zero_fixed:
             
            new_types_dict = {}
             
            for n, (t, v) in types.items():
                assert t != ExcitationType.VOLTAGE_FUN, "VOLTAGE_FUN excitation not supported for superposition."
                 
                if n == name:
                    new_types_dict[n] = (t, 1.0)
                elif t == ExcitationType.VOLTAGE_FIXED:
                    new_types_dict[n] = (t, 0.0)
                elif t == ExcitationType.CURRENT:
                    new_types_dict[n] = (t, 0.0)
                else:
                    new_types_dict[n] = (t, v)
            
            exc = Excitation(self.mesh)
            exc.excitation_types = new_types_dict
            excitations.append(exc)

        assert len(non_zero_fixed) == len(excitations)
        return {n:e for (n,e) in zip(non_zero_fixed, excitations)}

    def _get_active_elements(self, type_):
        assert type_ in ['electrostatic', 'magnetostatic']
        
        if self.mesh.symmetry == Symmetry.RADIAL:
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
    
    def _get_number_of_active_elements(self, type_):
        assert type_ in ['electrostatic', 'magnetostatic']
         
        if self.mesh.symmetry == Symmetry.RADIAL:
            elements = self.mesh.lines
            physicals = self.mesh.physical_to_lines
        else:
            elements = self.mesh.triangles
            physicals = self.mesh.physical_to_triangles
        
        def type_check(excitation_type):
            if type_ == 'electrostatic':
                return excitation_type.is_electrostatic()
            else:
                return excitation_type in [ExcitationType.MAGNETOSTATIC_POT, ExcitationType.MAGNETIZABLE]
         
        return sum(len(physicals[n]) for n, v in self.excitation_types.items() if type_check(v[0]))
    
    def get_electrostatic_active_elements(self):
        """Get elements in the mesh that are active, in the sense that
        an excitation to them has been applied. 
    
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
        """Get elements in the mesh that are active, in the sense that an excitation to them has been applied. 
    
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
     
    def get_number_of_electrostatic_active_elements(self):
        """Get elements in the mesh that are active, in the sense that
        an excitation to them has been applied. This is the length of the points
        array returned by the `Excitation.get_electrostatic_active_elements`.

        Returns
        --------
        int, giving the number of elements. """
        return self._get_number_of_active_elements('electrostatic')
    
    def get_number_of_electrostatic_matrix_elements(self):
        """Gets the number of elements along one axis of the matrix. If this function returns N, the
        matrix will have size NxN. The matrix consists of 64bit float values. Therefore the size of the matrix
        in bytes is 8·NxN.

        Returns
        ---------
        integer number
        """
        return self._get_number_of_active_elements('electrostatic')

        


