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
        with G.Geometry(G.Symmetry.RADIAL) as geom:
            
            circle1 = geom.add_circle([10e-3, 5e-3], 1.0e-3)
            circle2 = geom.add_circle([10e-3, -5e-3], 1.0e-3)
             
            geom.add_physical(circle1.plane_surface, 'coil1')
            geom.add_physical(circle2.plane_surface, 'coil2')
            
            geom.set_mesh_size_factor(20*MSF)
            mesh1 = geom.generate_triangle_mesh()

        with G.Geometry(G.Symmetry.RADIAL) as geom:
            rectangle = geom.add_rectangle(5e-3, 15e-3, -1e-3, 1e-3, 0)
            geom.add_physical(rectangle.curves, 'block')
            
            points = [[0, 50e-3], [100e-3, 50e-3], [100e-3, -50e-3], [0, -50e-3]]
            lines = [geom.add_line(geom.add_point(p1), geom.add_point(p2)) for p1, p2 in zip(points, points[1:])]
            geom.add_physical(lines, 'boundary')
             
            geom.set_mesh_size_factor(MSF)
            mesh2 = geom.generate_line_mesh(higher_order)
        
        return mesh1 + mesh2
    
    def supports_fmm(self):
        return False
    
    def supports_3d(self):
        return False
    
    def get_excitation(self, geometry):
        exc = E.Excitation(geometry)
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




