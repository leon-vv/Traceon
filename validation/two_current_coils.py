import os.path as path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

import traceon.geometry as G
import traceon.excitation as E
import traceon.plotting as P
import traceon.solver as S

from validation import Validation

class TwoCurrentCoils(Validation):
     
    def __init__(self):
        super().__init__('Calculate magnetic field of two current coils in presence of rectangular dielectric')
        self.plot_colors = dict(coil1='blue', coil2='red', block='orange')
    
    def create_mesh(self, MSF, symmetry, higher_order):
        boundary = G.Path.line([0., 0., 50e-3], [100e-3, 0., 50e-3]).line_to([100e-3, 0., -50e-3]).line_to([0., 0., -50e-3])
        boundary.name = 'boundary'

        coil1 = G.Surface.disk_xz(10e-3, 5e-3, 1e-3) 
        coil2 = G.Surface.disk_xz(10e-3, -5e-3, 1e-3) 

        coil1.name = 'coil1'
        coil2.name = 'coil2'

        coil_mesh = (coil1+coil2).mesh(mesh_size=0.25e-3)
        
        block  = G.Path.rectangle_xz(5e-3, 15e-3, -1e-3, 1e-3)
        block.name = 'block'

        return coil_mesh + (boundary + block).mesh(mesh_size_factor=MSF)
    
    def supports_fmm(self):
        return False
    
    def supports_3d(self):
        return False
    
    def get_excitation(self, mesh, symmetry):
        exc = E.Excitation(mesh, symmetry)
        exc.add_current(coil1= 10, coil2=-10)
        exc.add_magnetizable(block=25)
        exc.add_magnetostatic_boundary('boundary')
        return exc
    
    def correct_value_of_interest(self):
        return -91.94907464785867
    
    def compute_value_of_interest(self, geometry, field):
        return field.magnetostatic_field_at_point(np.array([10e-3, 0.]))[0]

if __name__ == '__main__':
    TwoCurrentCoils().run_validation()




