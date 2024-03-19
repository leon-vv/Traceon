import time, math

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pygmsh import *

import traceon.geometry as G
import traceon.excitation as E
import traceon.tracing as T
import traceon.solver as S

from validation import Validation


class SimpleMirror(Validation):

    def __init__(self):
        super().__init__('Calculated a trajectory through a simple flat mirror.')
        self.plot_colors = dict(mirror='brown', lens='blue', ground='green')

    def default_MSF(self, symmetry):
        if symmetry.is_3d():
            return [200, 400, 600, 1000, 1500]
        else:
            return [10, 50, 150, 250, 600]

    def create_mesh(self, MSF, symmetry, higher_order):
        _3d = symmetry.is_3d()
         
        if _3d:
            boundary = G.Path.line([0, 0, -1], [2, 0, -1]).line_to([2, 0, 1]).line_to([0.3, 0., 1]).revolve_z()
            mirror = G.Path.line([0., 0., 0.], [1., 0., 0.]).revolve_z()
        else:
            boundary = G.Path.line([0, -1], [2, -1]).line_to([2, 1]).line_to([0.3, 1])
            mirror = G.Path.line([0., 0.], [1., 0.])
         
        ms = 1/MSF
        return boundary.mesh(mesh_size=ms, name='ground', higher_order=higher_order) + \
                mirror.mesh(mesh_size=ms, name='mirror', higher_order=higher_order)
        
    def get_excitation(self, mesh, symmetry):
        excitation = E.Excitation(mesh, symmetry)
        excitation.add_voltage(mirror=-110, ground=0.0)
        return excitation
    
    def correct_value_of_interest(self): 
        return 1.6327355811e-01

    def compute_value_of_interest(self, geometry, field):
        _3d = geometry.is_3d()
        bounds = ((-0.22, 0.22), (-0.22, 0.22), (0.02, 11))
         
        axial_field = field.axial_derivative_interpolation(0.02, 4)
        tracer = T.Tracer(axial_field, bounds)
         
        pos = np.array([0.0, 10.0]) if not _3d else np.array([0.0, 0.0, 10.0])
        vel = T.velocity_vec_xz_plane(100, 1e-3, three_dimensional=_3d)
        
        st = time.time()
        _, pos = tracer(pos, vel)
        print(f'Trace took {(time.time()-st)*1000:.0f} ms')
        
        p = T.xy_plane_intersection(pos, 10)
        
        calculated = np.linalg.norm(p[:2] if _3d else p[:1])
        return calculated

if __name__ == '__main__':
    SimpleMirror().run_validation()


