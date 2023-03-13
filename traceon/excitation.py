from enum import IntEnum

import numpy as np

from .geometry import Symmetry

class ExcitationType(IntEnum):
    VOLTAGE_FIXED = 1
    VOLTAGE_FUN = 2
    DIELECTRIC = 3
    FLOATING_CONDUCTOR = 4

class Excitation:
     
    def __init__(self, mesh):
        self.mesh = mesh
        self.electrodes = mesh.get_electrodes()
        self.excitation_types = {}
     
    def add_voltage(self, **kwargs):
        for name, voltage in kwargs.items():
            assert name in self.electrodes
            if isinstance(voltage, int) or isinstance(voltage, float):
                self.excitation_types[name] = (ExcitationType.VOLTAGE_FIXED, voltage)
            elif callable(voltage):
                self.excitation_types[name] = (ExcitationType.VOLTAGE_FUN, voltage)
            else:
                raise NotImplementedError('Unrecognized voltage value')

    def add_dielectric(self, **kwargs):
        for name, permittivity in kwargs.items():
            assert name in self.electrodes
            self.excitation_types[name] = (ExcitationType.DIELECTRIC, permittivity)

    def add_floating_conductor(self, **kwargs):
        for name, charge in kwargs.items():
            assert name in self.electrodes
            self.excitation_types[name] = (ExcitationType.FLOATING_CONDUCTOR, charge)
     
    def _get_element_type(self):
        if self.mesh.symmetry == Symmetry.THREE_D:
            return 'triangle'
        else:
            return 'line'
    
    def get_floating_conductor_names(self):
        return [n for n, (t, v) in self.excitation_types.items() if t == ExcitationType.FLOATING_CONDUCTOR]
    
    def split_for_superposition(self):
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

    def get_active_vertices(self):
         
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
    
    def get_number_of_active_vertices(self):
        type_ = self._get_element_type()
        mesh = self.mesh.mesh
        
        return sum(len(mesh.cell_sets_dict[n][type_]) for n in self.excitation_types.keys())


        


