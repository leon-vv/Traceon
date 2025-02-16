import time, math

import numpy as np
from scipy.interpolate import CubicSpline

import voltrace as v
from validation import Validation

try:
    from voltrace_pro.field import Field3DAxial
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
        boundary = v.Path.line([0, 0, -1], [2, 0, -1]).extend_with_line([2, 0, 1]).extend_with_line([0.3, 0., 1])
        mirror = v.Path.line([0., 0., 0.], [1., 0., 0.])

        boundary.name = 'boundary'
        mirror.name = 'mirror'
        
        _3d = symmetry.is_3d()
         
        if symmetry.is_3d():
            boundary = boundary.revolve_z()
            mirror = mirror.revolve_z()
        
        return (boundary + mirror).mesh(mesh_size_factor=MSF)
     
    def get_excitation(self, mesh, symmetry):
        excitation = v.Excitation(mesh, symmetry)
        excitation.add_voltage(mirror=-110, boundary=0.0)
        return excitation
    
    def correct_value_of_interest(self): 
        return 1.6327355811e-01

    def compute_value_of_interest(self, geometry, field):
        _3d = geometry.is_3d()
        assert not _3d or v.Field3DAxial is not None, "Please install voltrace_pro for fast 3D tracing support"
        
        bounds = ((-0.22, 0.22), (-0.22, 0.22), (0.02, 11))
         
        axial_field = v.FieldRadialAxial(field, 0.02, 4) if not _3d else v.Field3DAxial(field, 0.02, 4)
        tracer = axial_field.get_tracer(bounds)
         
        pos = [0.0, 0.0, 10.0]
        vel = v.velocity_vec_xz_plane(100, 1e-3)
        
        st = time.time()
        trajectory = tracer(pos, vel)
        print(f'Trace took {(time.time()-st)*1000:.0f} ms')
        
        intersection_time = trajectory.xy_plane_intersection(10)
        p = trajectory(intersection_time)
        return np.linalg.norm(p[:2])

if __name__ == '__main__':
    SimpleMirror().run_validation()


