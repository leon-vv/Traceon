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
        boundary = G.Path.line([0., 0., 5.], [5., 0., 5.]).line_to([5., 0., 0]).line_to([0., 0., 0.]);
        boundary.name = 'boundary'
        mesh1 = boundary.mesh(mesh_size_factor=MSF)

        coil = G.Surface.rectangle_xz(2., 3., 2., 3.)
        coil.name = 'coil'
        mesh2 = coil.mesh(mesh_size_factor=MSF)
        return mesh1 + mesh2

    def supports_3d(self):
        return False
    
    def get_excitation(self, mesh, symmetry):
        exc = E.Excitation(mesh, symmetry)
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

