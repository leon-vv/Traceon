
from enum import IntEnum

import numpy as np

class ExcitationType(IntEnum):
    VOLTAGE_FIXED = 1
    VOLTAGE_FUN = 2
    DIELECTRIC = 3

class Excitation:
     
    def __init__(self, geom):
        self.geometry = geom
        self.electrodes = geom.get_electrodes()
        self.excitation_types = {}
     
    def add_voltage(self, **kwargs):
        for name, voltage in kwargs.items():
            assert name in self.electrodes
            if isinstance(voltage, int):
                self.excitation_types[name] = (ExcitationType.VOLTAGE_FIXED, voltage)
            elif callable(voltage):
                self.excitation_types[name] = (ExcitationType.VOLTAGE_FUN, voltage)
            else:
                raise NotImplementedError('Unrecognized voltage value')

    def add_dielectric(self, **kwargs):
        for name, permittivity in kwargs.items():
            assert name in self.electrodes
            self.excitation_types[name] = (ExcitationType.DIELECTRIC, permittivity)
     
    def get_active_lines(self):
        
        mesh = self.geometry.mesh
        lines = mesh.cells_dict['line']
        inactive = np.full(len(lines), True)
        names = {}
        
        for name in self.excitation_types.keys():
            inactive[ mesh.cell_sets_dict[name]['line'] ] = False
        
        map_index = np.arange(len(lines)) - np.cumsum(inactive)
        
        for name in self.excitation_types.keys():
            names[name] = map_index[mesh.cell_sets_dict[name]['line']]
              
        line_points = mesh.points[ lines[~inactive] ]
        
        return line_points, names


        


