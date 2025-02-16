import time, math

import numpy as np
from scipy.interpolate import CubicSpline

import traceon.geometry as G
import traceon.excitation as E
import traceon.tracing as T
import traceon.solver as S
from traceon.field import FieldRadialAxial

from validation import Validation

try:
    from traceon_pro.field import Field3DAxial
except ImportError:
    Field3DAxial = None


class SimpleMirror(Validation):

    def __init__(self):
        super().__init__('Calculated a trajectory through a simple flat mirror.')
        self.plot_colors = dict(mirror='brown', lens='blue', boundary='green')

    def default_MSF(self, symmetry):
        if not symmetry.is_3d():
            return [8, 16, 32, 64]
        else:
            return [4, 8, 16, 32]
    
    def create_mesh(self, MSF, symmetry, higher_order):
        boundary = G.Path.line([0, 0, -1], [2, 0, -1]).extend_with_line([2, 0, 1]).extend_with_line([0.3, 0., 1])
        mirror = G.Path.line([0., 0., 0.], [1., 0., 0.])

        boundary.name = 'boundary'
        mirror.name = 'mirror'
        
        _3d = symmetry.is_3d()
         
        if symmetry.is_3d():
            boundary = boundary.revolve_z()
            mirror = mirror.revolve_z()
        
        return (boundary + mirror).mesh(mesh_size_factor=MSF)
     
    def get_excitation(self, mesh, symmetry):
        excitation = E.Excitation(mesh, symmetry)
        excitation.add_voltage(mirror=-110, boundary=0.0)
        return excitation
    
    def correct_value_of_interest(self): 
        return 1.6327355811e-01

    def compute_value_of_interest(self, geometry, field):
        _3d = geometry.is_3d()
        assert not _3d or Field3DAxial is not None, "Please install traceon_pro for fast 3D tracing support"
        
        bounds = ((-0.22, 0.22), (-0.22, 0.22), (0.02, 11))
         
        axial_field = FieldRadialAxial(field, 0.02, 4) if not _3d else Field3DAxial(field, 0.02, 4)
        tracer = axial_field.get_tracer(bounds)
         
        pos = [0.0, 0.0, 10.0]
        vel = T.velocity_vec_xz_plane(100, 1e-3)
        
        st = time.time()
        _, pos = tracer(pos, vel)
        print(f'Trace took {(time.time()-st)*1000:.0f} ms')
        
        p = T.xy_plane_intersection(pos, 10)
        
        calculated = np.linalg.norm(p[:2] if _3d else p[:1])
        return calculated

if __name__ == '__main__':
    SimpleMirror().run_validation()


