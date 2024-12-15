import time

import numpy as np

import traceon.geometry as G
import traceon.excitation as E
import traceon.tracing as T
import traceon.solver as S
from traceon.interpolation import FieldRadialAxial

from validation import Validation

try:
    from traceon_pro.interpolation import Field3DAxial
except ImportError:
    Field3DAxial = None

class DohiMirror(Validation):

    def __init__(self):
        super().__init__('''
            Consider the accuracy of electron tracing using a field calculated by a radial series expansion
            using the axial derivatives. The accuracy of the trace is determined by computing the r value the
            electron has at z0=15mm after reflection of the Dohi mirror, see:

            H. Dohi, P. Kruit. Design for an aberration corrected scanning electron microscope using
            miniature electron mirrors. 2018.
            ''')
        
        self.plot_colors = dict(mirror='brown', lens='blue', ground='green', boundary='purple')


    def create_mesh(self, MSF, symmetry, higher_order):
             
        rmax = 1.0
        margin = 0.3
        extent = rmax-0.1

        t = 0.15 # thickness
        r = 0.075 # radius
        st = 0.5  # spacer thickness

        mirror = G.Path.aperture(0.15, r, extent, z=t/2)
        mirror.name = 'mirror'
        
        mirror_line = G.Path.line([0., 0., 0.], [r, 0., 0.])
        mirror_line.name = 'mirror'

        lens = G.Path.aperture(0.15, r, extent, z=t + st + t/2)
        lens.name = 'lens'
        
        ground = G.Path.aperture(0.15, r, extent, z=t+st+t+st+t/2)
        ground.name = 'ground'
    
        boundary = G.Path.line([0., 0., 1.75], [rmax, 0., 1.75]) \
            .extend_with_line([rmax, 0., -0.3]).extend_with_line([0., 0., -0.3])
        boundary.name = 'boundary'
        
        geom = mirror+mirror_line+lens+ground+boundary
         
        if symmetry.is_3d():
            geom = geom.revolve_z()
        
        return geom.mesh(mesh_size_factor=MSF)
     
    def get_excitation(self, mesh, symmetry):
        exc = E.Excitation(mesh, symmetry)
        exc.add_voltage(ground=0.0, mirror=-1250, lens=710.0126605741955)
        exc.add_electrostatic_boundary('boundary')
        return exc

    def correct_value_of_interest(self):
        return 0. # Determined by a accurate, not interpolated trace

    def compute_accuracy(self, computed, correct):
        return abs(computed)
    
    def compute_value_of_interest(self, mesh, field):
        _3d = mesh.is_3d()
        assert not _3d or Field3DAxial is not None, "Please install traceon_pro for fast 3D tracing support"
        
        axial_field = FieldRadialAxial(field, 0.05, 1.7, 500) if not _3d else Field3DAxial(field, 0.05, 1.7, 500)
        
        bounds = ((-0.1, 0.1), (-0.1, 0.1), (0.05, 1.7))
        field.set_bounds(bounds)
        
        bounds = ((-0.1, 0.1), (-0.03, 0.03), (0.05, 19.0))
        tracer_derivs = axial_field.get_tracer(bounds)
        
        angle = 0.5e-3
        z0 = 15
        
        start_pos = np.array([0.0, 0.0, z0])
        start_vel = T.velocity_vec_xz_plane(1000, angle)
         
        print('Starting trace...')
        st = time.time()
        _, pos_derivs = tracer_derivs(start_pos, start_vel)
        print(f'Trace took {(time.time()-st)*1000:.1f} ms')
         
        intersection = T.xy_plane_intersection(pos_derivs, z0)
        return intersection[0]

if __name__ == '__main__':
    '''
    from scipy.optimize import newton
    dohi = DohiMirror()
    geom = dohi.create_mesh(200, G.Symmetry.RADIAL)
    exc = dohi.get_excitation(geom)
    exc.add_voltage(lens=720)
    fields = S.solve_direct(exc, superposition=True)

    def opt(lens_voltage):
        field = -1250*fields['mirror'] + lens_voltage*fields['lens']
        intersection = dohi.compute_value_of_interest(geom, field)
        print(intersection)
        return intersection
    
    print(newton(opt, 710.0126605741955, tol=1e-9))
    '''
    DohiMirror().run_validation()



