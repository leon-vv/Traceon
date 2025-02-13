import os.path as path

import numpy as np
from scipy.interpolate import CubicSpline

import voltrace as T
from validation import Validation

class TwoCurrentCoils(Validation):
     
    def __init__(self):
        super().__init__('Calculate magnetic field of two current coils in presence of rectangular dielectric')
        self.plot_colors = dict(coil1='blue', coil2='red', block='orange')
    
    def create_mesh(self, MSF, symmetry, higher_order):
        boundary = T.Path.line([0., 0., 50e-3], [100e-3, 0., 50e-3]).extend_with_line([100e-3, 0., -50e-3]).extend_with_line([0., 0., -50e-3])
        boundary.name = 'boundary'

        coil1 = T.Surface.disk_xz(10e-3, 5e-3, 1e-3) 
        coil2 = T.Surface.disk_xz(10e-3, -5e-3, 1e-3) 

        coil1.name = 'coil1'
        coil2.name = 'coil2'

        coil_mesh = (coil1+coil2).mesh(mesh_size=0.25e-3)
        
        block  = T.Path.rectangle_xz(5e-3, 15e-3, -1e-3, 1e-3)
        block.name = 'block'

        return coil_mesh + (boundary + block).mesh(mesh_size_factor=MSF)
    
    def supports_fmm(self):
        return False
    
    def supports_3d(self):
        return False
    
    def get_excitation(self, mesh, symmetry):
        exc = T.Excitation(mesh, symmetry)
        exc.add_current(coil1= 10, coil2=-10)
        exc.add_magnetizable(block=25)
        exc.add_magnetostatic_boundary('boundary')
        return exc
    
    def correct_value_of_interest(self):
        return -91.94907464785867
    
    def compute_value_of_interest(self, geometry, field):
        return field.magnetostatic_field_at_point([10e-3, 0., 0.])[0]

if __name__ == '__main__':
    TwoCurrentCoils().run_validation()




