import time

import numpy as np

import traceon.geometry as G
import traceon.solver as S
import traceon.excitation as E
import traceon.plotting as P
import traceon.tracing as T
from traceon.field import FieldRadialAxial

from validation import Validation

try:
    from traceon_pro.field import Field3DAxial
except ImportError:
    Field3DAxial = None


THICKNESS = 0.5
SPACING = 0.5
RADIUS = 0.15
z0 = -THICKNESS - SPACING - THICKNESS/2 - 0.5

class MagneticEinzelLens(Validation):

    def __init__(self):
        super().__init__('Calculate focal length of an einzel lens.')
        self.plot_colors = dict(lens='blue', ground='green', boundary='purple')

    def create_mesh(self, MSF, symmetry, higher_order):
        boundary = G.Path.line([0., 0., 1.75],  [2.0, 0., 1.75])\
            .extend_with_line([2.0, 0., -1.75]).extend_with_line([0., 0., -1.75])
        
        margin_right = 0.1
        extent = 2.0 - margin_right
        
        bottom = G.Path.aperture(THICKNESS, RADIUS, extent, -THICKNESS - SPACING)
        middle = G.Path.aperture(THICKNESS, RADIUS, extent)
        top = G.Path.aperture(THICKNESS, RADIUS, extent, THICKNESS + SPACING)
         
        if symmetry.is_3d():
            boundary = boundary.revolve_z()
            bottom = bottom.revolve_z()
            middle = middle.revolve_z()
            top = top.revolve_z()

        boundary.name = 'boundary'
        bottom.name = 'ground'
        middle.name = 'lens'
        top.name = 'ground'
        
        return (boundary + bottom + middle + top).mesh(mesh_size_factor=MSF)


    def get_excitation(self, mesh, symmetry):
        excitation = E.Excitation(mesh, symmetry)
        excitation.add_magnetostatic_potential(ground=0.0, lens=50.)
        excitation.add_magnetostatic_boundary('boundary')
        return excitation

    def supports_fmm(self):
        return False
     
    def correct_value_of_interest(self):
        return 4.08641734
      
    def compute_value_of_interest(self, geom, field):
        _3d = geom.is_3d()
        assert not _3d or Field3DAxial is not None, "Please install traceon_pro for fast 3D tracing support"

        field.set_bounds( ((-RADIUS, RADIUS), (-RADIUS, RADIUS), (-1.5,1.5)) )
        field_axial = FieldRadialAxial(field, -1.5, 1.5, 1000) if not _3d else Field3DAxial(field, -1.5, 1.5, 1000)
         
        bounds = ((-RADIUS, RADIUS), (-RADIUS, RADIUS), (-5, 3.5))
        tracer = field_axial.get_tracer(bounds)
        
        p0 = np.array([RADIUS/5, 0.0, 3])
        v0 = T.velocity_vec_xz_plane(1000, 0)
          
        st = time.time()
        _, pos = tracer(p0, v0)
        
        return -T.axis_intersection(pos)

if __name__ == '__main__':
    MagneticEinzelLens().run_validation()

