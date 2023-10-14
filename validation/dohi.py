import time

import numpy as np
import matplotlib.pyplot as plt

import traceon.geometry as G
import traceon.excitation as E
import traceon.tracing as T
import traceon.solver as S

from validation import Validation


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


    def create_mesh(self, MSF, symmetry):
            
        revolve_factor = 0.0 if symmetry == G.Symmetry.RADIAL else 1.0
        rmax = 1.0
        margin = 0.3
        
        with G.MEMSStack(z0=-margin, margin=margin, zmin=0.1, zmax=1.7, size_from_distance=True, revolve_factor=revolve_factor, rmax=rmax) as geom:
            
            # Close mirror at the bottom, like in the paper
            if symmetry == G.Symmetry.THREE_D_HIGHER_ORDER:
                mirror_line = geom.add_line(geom.add_point([0.0, 0.0, 0.0]), geom.add_point([0.075, 0.0, 0.0]))
                revolved = G.revolve_around_optical_axis(geom, [mirror_line], revolve_factor)
                geom.add_physical(revolved, 'mirror')
            else:
                mirror_line = geom.add_line(geom.add_point([0.0, 0.0]), geom.add_point([0.075, 0.0]))
                geom.add_physical(mirror_line, 'mirror')
            
            geom.add_electrode(0.075, 0.150, 'mirror')
            geom.add_spacer(0.5)
            geom.add_electrode(0.075, 0.150, 'lens')
            geom.add_spacer(0.5)
            geom.add_electrode(0.075, 0.150, 'ground')
            
            geom.set_mesh_size_factor(MSF)
            
            return geom.generate_mesh()

    def get_excitation(self, mesh):
        exc = E.Excitation(mesh)
        exc.add_voltage(ground=0.0, mirror=-1250, lens=710)
        exc.add_boundary('boundary')
        return exc

    def correct_value_of_interest(self):
        return 3.13452443471595e-03 # Determined by a accurate, not interpolated trace
    
    def compute_value_of_interest(self, geom, field):
        axial_field = field.axial_derivative_interpolation(0.05, 1.7, 500)
         
        bounds = ((-0.1, 0.1), (-0.1, 0.1), (0.05, 1.7)) if geom.symmetry == G.Symmetry.THREE_D_HIGHER_ORDER else ((-0.1, 0.1), (0.05, 1.7))
        field.set_bounds(bounds)
        
        bounds = ((-0.1, 0.1), (-0.03, 0.03), (0.05, 19.0)) if geom.symmetry == G.Symmetry.THREE_D_HIGHER_ORDER else ((-0.03, 0.03), (0.05, 19.0))
        tracer_derivs = T.Tracer(axial_field, bounds)
        
        angle = 0.5e-3
        z0 = 15
        
        start_pos = np.array([0.0, 0.0, z0]) if geom.symmetry == G.Symmetry.THREE_D_HIGHER_ORDER else np.array([0.0, z0])
        start_vel = T.velocity_vec_xz_plane(1000, angle, three_dimensional=geom.symmetry == G.Symmetry.THREE_D_HIGHER_ORDER)
         
        print('Starting trace...')
        st = time.time()
        _, pos_derivs = tracer_derivs(start_pos, start_vel)
        print(f'Trace took {(time.time()-st)*1000:.1f} ms')
         
        intersection = T.xy_plane_intersection(pos_derivs, z0)
        return intersection[0]

if __name__ == '__main__':
    DohiMirror().run_validation()
