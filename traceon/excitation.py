"""The excitation module allows to specify the excitation (or element types) of the different physical groups (electrodes)
created with the `traceon.geometry` module. Currently only electrostatic excitations are supported.

The possible excitations are as follows:

- Fixed voltage (electrode connect to a power supply)
- Voltage function (a generic Python function specifies the voltage as a function of position)
- Dielectric, with arbitrary electric permittivity
- Floating conductor, with an arbitrary total charge on the surface

Once the excitation is specified, it can be passed to `traceon.solver.solve_bem` to compute the resulting field.
"""


from enum import IntEnum

import numpy as np

from .geometry import Symmetry

class ExcitationType(IntEnum):
    """Possible excitation that can be applied to elements of the geometry. See the methods of `Excitation` for documentation."""
    VOLTAGE_FIXED = 1
    VOLTAGE_FUN = 2
    DIELECTRIC = 3
    FLOATING_CONDUCTOR = 4

    def __str__(self):
        if self == ExcitationType.VOLTAGE_FIXED:
            return 'voltage fixed'
        elif self == ExcitationType.VOLTAGE_FUN:
            return 'voltage function'
        elif self == ExcitationType.DIELECTRIC:
            return 'dielectric'
        elif self == ExcitationType.FLOATING_CONDUCTOR:
            return 'floating conductor'


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

    def add_floating_conductor(self, **kwargs):
        """
        Specify geometric elements as floating conductors, and specify the total charge on the conductor.
         
        Parameters
        ----------
        **kwargs : dict
            The keys of the dictionary are the geometry names, while the values are the charge on the conductors. For example,
            calling the function as `add_floating_conductor(spacer=10)` specifies the physical group `spacer` as a floating conductor with
            a total charge on its surface equal to 10. For the unit of charge, see the section 'Units' on the `traceon` page.
            
        """
        for name, charge in kwargs.items():
            assert name in self.electrodes
            self.excitation_types[name] = (ExcitationType.FLOATING_CONDUCTOR, charge)
     
    def _get_element_type(self):
        if self.mesh.symmetry == Symmetry.THREE_D:
            return 'triangle'
        else:
            return 'line'
    
        
    def _split_for_superposition(self):
        
        # Names that have a fixed voltage excitation, not equal to 0.0
        types = self.excitation_types
        non_zero_fixed = [n for n, (t, v) in types.items() if t == ExcitationType.VOLTAGE_FIXED and v != 0.0]
        
        excitations = []
         
        for name in non_zero_fixed:

            new_types_dict = {}
             
            for n, (t, v) in types.items():
                assert t != ExcitationType.VOLTAGE_FUN, "VOLTAGE_FUN excitation not supported for superposition."
                assert (t != ExcitationType.FLOATING_CONDUCTOR or v == 0.0), "FLOATING_CONDUCTOR only supported in superposition if total charge equals zero."
                 
                if n == name:
                    new_types_dict[n] = (t, 1.0)
                elif t == ExcitationType.VOLTAGE_FIXED:
                    new_types_dict[n] = (t, 0.0)
                else:
                    new_types_dict[n] = (t, v)
            
            exc = Excitation(self.mesh)
            exc.excitation_types = new_types_dict
            excitations.append(exc)

        assert len(non_zero_fixed) == len(excitations)
        return {n:e for (n,e) in zip(non_zero_fixed, excitations)}

    def get_active_elements(self):
        """Get elements in the mesh that are active, in the sense that
        an excitation to them has been applied. 
    
        Returns
        --------
        A tuple of two elements: (points, names). points is a Numpy array of shape (N, 2, 3) in the case of 2D and (N, 3, 3) in the case of 3D. \
        This array contains the vertices of the line elements or the triangles. \
        names is a dictionary, the keys being the names of the physical groups mentioned by this excitation, \
        while the values are Numpy arrays of indices that can be used to index the points array.
        """
        type_ = self._get_element_type()
        mesh = self.mesh.mesh
        vertices = mesh.cells_dict[type_] # Indices making up the lines and triangles
        inactive = np.full(len(vertices), True)
        names = {}
        
        for name in self.excitation_types.keys():
            inactive[ mesh.cell_sets_dict[name][type_] ] = False
        
        map_index = np.arange(len(vertices)) - np.cumsum(inactive)
        
        for name in self.excitation_types.keys():
            names[name] = map_index[mesh.cell_sets_dict[name][type_]]
              
        return mesh.points[ vertices[~inactive] ], names
    
    def get_number_of_active_elements(self):
        """Get elements in the mesh that are active, in the sense that
        an excitation to them has been applied. This is the length of the points
        array returned by the `Excitation.get_active_elements`.

        Returns
        --------
        int, giving the number of elements. """
        type_ = self._get_element_type()
        mesh = self.mesh.mesh
        
        return sum(len(mesh.cell_sets_dict[n][type_]) for n in self.excitation_types.keys())


        


