import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import mu_0

import traceon.geometry as G
import traceon.solver as S
import traceon.excitation as E
import traceon.plotting as P
import traceon.tracing as T

from validation import Validation

class RectangularCoil(Validation):

    def __init__(self):
        super().__init__('Calculate current field of rectangular coil.')
        self.plot_colors = dict(coil='red', boundary='purple')

    def create_mesh(self, MSF, symmetry, higher_order):
        assert symmetry == G.Symmetry.RADIAL, "3D meshes not yet supported in magnetostatics"
         
        with G.Geometry(G.Symmetry.RADIAL) as geom:
            points = [[0, 5], [5,5], [5,0], [0, 0]]
            lines = [geom.add_line(geom.add_point(p1), geom.add_point(p2)) for p1, p2 in zip(points, points[1:])]
            geom.add_physical(lines, 'boundary')
             
            #circle = geom.add_circle([2.5, 4], 0.5)
            #geom.add_physical(circle.curve_loop.curves, 'circle')
            
            geom.set_mesh_size_factor(MSF)
            mesh1 = geom.generate_line_mesh(higher_order)

        with G.Geometry(G.Symmetry.RADIAL) as geom:
            rect = geom.add_rectangle(2, 3, 2, 3, 0)
            geom.add_physical(rect.surface, 'coil')
            geom.set_mesh_size_factor(MSF)
            mesh2 = geom.generate_triangle_mesh()

        return mesh1 + mesh2

    def supports_3d(self):
        return False
    
    def get_excitation(self, mesh):
        exc = E.Excitation(mesh)
        exc.add_current(coil=1)
        exc.add_magnetostatic_boundary('boundary')
        return exc
     
    def correct_value_of_interest(self):
        cr, cz = 0.001152863994, 2.647794398E-4 # Correct
        return cr/(1e4*mu_0)
     
    def compute_value_of_interest(self, geom, field):
        fr, fz = field.magnetostatic_field_at_point(np.array([2.5, 4]))
        return fr

if __name__ == '__main__':
    RectangularCoil().run_validation()

