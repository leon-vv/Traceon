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

class RectangularCoilWithCircle(Validation):

    def __init__(self):
        super().__init__('Calculate current field of rectangular coil in presence of dielectric torus.')
        self.plot_colors = dict(coil='red', boundary='purple')

    def create_mesh(self, MSF, symmetry, higher_order):
        boundary = G.Path.line([0., 0., 5.], [5., 0., 5.]).line_to([5., 0., 0]).line_to([0., 0., 0.]);
        boundary.name = 'boundary'

        circle = G.Path.circle_xz(2.5, 4, 0.5)
        circle.name = 'circle'
        mesh1 = (boundary + circle).mesh(mesh_size_factor=MSF, higher_order=higher_order)
        
        coil = G.Surface.rectangle_xz(2., 3., 2., 3.)
        coil.name = 'coil'
        mesh2 = coil.mesh(mesh_size=0.1)
        
        return mesh1 + mesh2
    
    def supports_3d(self):
        return False
     
    def get_excitation(self, mesh, symmetry):
        exc = E.Excitation(mesh, symmetry)
        exc.add_current(coil=1)
        exc.add_magnetostatic_boundary('boundary')
        exc.add_magnetizable(circle=10)
        return exc
     
    def correct_value_of_interest(self):
        cr, cz = 154.54131081, 37.39247774 # Correct
        return cr/1e4
     
    def compute_value_of_interest(self, geom, field):
        fr, fz = field.magnetostatic_field_at_point(np.array([2.5, 4]))
        return fr

if __name__ == '__main__':
    RectangularCoilWithCircle().run_validation()

