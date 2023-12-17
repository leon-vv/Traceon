import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.excitation as E
import traceon.plotting as P
import traceon.solver as S

from validation import Validation

class MagneticTorus(Validation):
     
    def __init__(self):
        super().__init__()
        self.plot_colors = dict(boundary='blue', coil='red', circle='green')
    
    def create_mesh(self, MSF, symmetry, higher_order):
        with G.Geometry(G.Symmetry.RADIAL) as geom:
            points = [[0, 5], [5,5], [5,0], [0, 0]]
            lines = [geom.add_line(geom.add_point(p1), geom.add_point(p2)) for p1, p2 in zip(points, points[1:])]
            geom.add_physical(lines, 'boundary')
            
            circle = geom.add_circle([2.5, 4], 0.5)
            geom.add_physical(circle.curve_loop.curves, 'circle')
            
            geom.set_mesh_size_factor(MSF)
            mesh1 = geom.generate_line_mesh(True)
        
        with G.Geometry(G.Symmetry.RADIAL) as geom:
            rect = geom.add_rectangle(2, 3, 2, 3, 0)
            geom.add_physical(rect.surface, 'coil')
             
            geom.set_mesh_size_factor(MSF)
            mesh2 = geom.generate_triangle_mesh(True)

        return mesh1 + mesh2
    
    def supports_fmm(self):
        return False
    
    def get_excitation(self, geometry):
        exc = E.Excitation(geometry)
        exc.add_current(coil=1)
        exc.add_magnetizable(circle=10)
        exc.add_boundary('boundary')
        return exc
    
    def correct_value_of_interest(self):
        return 0.15454131081
    
    def compute_value_of_interest(self, geometry, field):
        return field.magnetostatic_field_at_point(np.array([2.5, 4]))[0]

if __name__ == '__main__':
    MagneticTorus().run_validation()




