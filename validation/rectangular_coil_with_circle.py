import time

import numpy as np
from scipy.constants import mu_0

import voltrace as T
from validation import Validation

class RectangularCoilWithCircle(Validation):

    def __init__(self):
        super().__init__('Calculate current field of rectangular coil in presence of dielectric torus.')
        self.plot_colors = dict(coil='red', boundary='purple')

    def create_mesh(self, MSF, symmetry, higher_order):
        boundary = T.Path.line([0., 0., 5.], [5., 0., 5.]).extend_with_line([5., 0., 0]).extend_with_line([0., 0., 0.]);
        boundary.name = 'boundary'

        circle = T.Path.circle_xz(2.5, 4, 0.5)
        circle.name = 'circle'
        mesh1 = (boundary + circle).mesh(mesh_size_factor=MSF, higher_order=higher_order)
        
        coil = T.Surface.rectangle_xz(2., 3., 2., 3.)
        coil.name = 'coil'
        mesh2 = coil.mesh(mesh_size=0.1)
        
        return mesh1 + mesh2
    
    def supports_3d(self):
        return False
     
    def get_excitation(self, mesh, symmetry):
        exc = T.Excitation(mesh, symmetry)
        exc.add_current(coil=1)
        exc.add_magnetostatic_boundary('boundary')
        exc.add_magnetizable(circle=10)
        return exc
     
    def correct_value_of_interest(self):
        cr, cz = 154.54131081, 37.39247774 # Correct
        return cr/1e4
     
    def compute_value_of_interest(self, geom, field):
        fx, _, fz = field.magnetostatic_field_at_point([2.5, 0.0, 4])
        return fx

if __name__ == '__main__':
    RectangularCoilWithCircle().run_validation()

