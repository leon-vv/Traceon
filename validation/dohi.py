import time

import numpy as np
import matplotlib.pyplot as plt

import traceon.geometry as G
import traceon.excitation as E
import traceon.tracing as T
import traceon.solver as S

import util

def create_geometry(N, symmetry, for_plot):
    
    MESH_SIZE = 1/N

    revolve_factor = 0.0

    bounds = ((-0.03, 0.03), (0.1, 5.0))

    if symmetry == '3d':
        bounds = (bounds[0], (-0.03, 0.03), bounds[1])
        revolve_factor = 1.0 if not for_plot else 0.6
        rmax = 1.25
    else:
        rmax = 1.25
    
    with G.MEMSStack(bounds, z0=-0.3-0.15, revolve_factor=revolve_factor, rmax=rmax, mesh_size=MESH_SIZE) as geom:
        
        # Close mirror at the bottom, like in the paper
        if symmetry == '3d':
            mirror_line = geom.add_line(geom.add_point([0.0, 0.0, 0.0], MESH_SIZE), geom.add_point([0.075, 0.0, 0.0], MESH_SIZE))
            revolved = G.revolve_around_optical_axis(geom, [mirror_line], revolve_factor)
            geom._add_physical('mirror', revolved)
        else:
            mirror_line = geom.add_line(geom.add_point([0.0, 0.0], MESH_SIZE), geom.add_point([0.075, 0.0], MESH_SIZE))
            geom._add_physical('mirror', [mirror_line])
        
        geom.add_electrode(0.075, 0.150, 'ground')
        geom.add_spacer(0.3)
        geom.add_electrode(0.075, 0.150, 'mirror')
        geom.add_spacer(0.5)
        geom.add_electrode(0.075, 0.150, 'lens')
        geom.add_spacer(0.5)
        geom.add_electrode(0.075, 0.150, 'ground')

        mesh = geom.generate_mesh(dim=1 if symmetry != '3d' else 2)
    
         
    return G.Geometry(mesh, 1/MESH_SIZE, bounds, symmetry=symmetry)

def compute_error(geom):
    exc = E.Excitation(geom)
    exc.add_voltage(ground=0.0, mirror=-1250, lens=695)
    
    field = S.solve_bem(exc)

    axial_field = field.axial_derivative_interpolation(0.1, 5.0)
     
    bounds = ((-0.03, 0.03), (-0.03, 0.03), (0.05, 19.0)) if geom.symmetry == '3d' else ((-0.03, 0.03), (0.05, 19.0))
    tracer_derivs = T.Tracer(axial_field, bounds)
    
    angle = 0.5e-3
    z0 = 15
    
    start_pos = np.array([0.0, 0.0, z0]) if geom.symmetry == '3d' else np.array([0.0, z0])
    start_vel = T.velocity_vec_xz_plane(1000, angle, three_dimensional=geom.symmetry == '3d')
     
    print('Starting trace.')
    st = time.time()
    _, pos_derivs = tracer_derivs(start_pos, start_vel)
    print(f'Trace took {(time.time()-st)*1000:.1f} ms')
     
    correct = 3.12936530852257e-03 # Determined by a accurate, naive trace
    int_derivs = T.xy_plane_intersection(pos_derivs, z0)

    print(f'Calculated intersection: {int_derivs[0]:.14e} mm (correct: {correct:.4e} mm)')
     
    return exc.get_number_of_active_vertices(), abs(int_derivs[0]/correct - 1)


util.parser.description = '''
Consider the accuracy of electron tracing using a field calculated by a radial series expansion
using the axial derivatives. The accuracy of the trace is determined by computing the r value the
electron has at z0=15mm after reflection of the Dohi mirror, see:

H. Dohi, P. Kruit. Design for an aberration corrected scanning electron microscope using
miniature electron mirrors. 2018.
'''
util.parse_validation_args(create_geometry, compute_error, mirror='brown', lens='blue', ground='green',
    N={'radial': [50, 100, 200, 300], '3d': [5, 10, 20]})


