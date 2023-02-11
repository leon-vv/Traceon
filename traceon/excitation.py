
from enum import IntEnum

import numpy as np

class ExcitationType(IntEnum):
    VOLTAGE_FIXED = 1
    VOLTAGE_FUN = 2
    DIELECTRIC = 3
    FLOATING_CONDUCTOR = 4

class Excitation:
     
    def __init__(self, geom):
        self.geometry = geom
        self.electrodes = geom.get_electrodes()
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
     
    def get_active_vertices(self):

        if self.geometry.symmetry == '3d':
            type_ = 'triangle'
        else:
            type_ = 'line'
        
        mesh = self.geometry.mesh
        vertices = mesh.cells_dict[type_] # Indices making up the lines and triangles
        inactive = np.full(len(vertices), True)
        names = {}
        
        for name in self.excitation_types.keys():
            inactive[ mesh.cell_sets_dict[name][type_] ] = False
        
        map_index = np.arange(len(vertices)) - np.cumsum(inactive)
        
        for name in self.excitation_types.keys():
            names[name] = map_index[mesh.cell_sets_dict[name][type_]]
              
        return mesh.points[ vertices[~inactive] ], names
    
    def get_number_of_active_lines(self):
        mesh = self.geometry.mesh
        return sum(len(mesh.cell_sets_dict[n]['line']) for n in self.excitation_types.keys())


        


