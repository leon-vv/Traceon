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
    assert symmetry == 'radial', 'Only radial symmetry supported'
    
    with G.MEMSStack(z0=-0.3-0.15, rmax=1.25, margin_right=0.2, mesh_size=MESH_SIZE) as geom:
        
        # Close mirror at the bottom, like in the paper
        mirror_line = geom.add_line(geom.add_point([0.0, 0.0], MESH_SIZE), geom.add_point([0.075, 0.0], MESH_SIZE))
        geom._add_physical('mirror', [mirror_line])
        
        geom.add_electrode(0.075, 0.150, 'ground')
        geom.add_spacer(0.3)
        geom.add_electrode(0.075, 0.150, 'mirror')
        geom.add_spacer(0.5)
        geom.add_electrode(0.075, 0.150, 'lens')
        geom.add_spacer(0.5)
        geom.add_electrode(0.075, 0.150, 'ground')

        mesh = geom.generate_mesh(dim=1)

    bounds = ((-0.03, 0.03), (0.1, 2.0))
    return G.Geometry(mesh, 1/MESH_SIZE, bounds)

def compute_error(geom):
    exc = E.Excitation(geom)
    exc.add_voltage(ground=0.0, mirror=-2139.1093, lens=-1200)
    
    field = S.solve_bem(exc)
     
    tracer_hermite = T.Tracer(field, ( (-0.03, 0.03), (0.05, 19.0) ), T.Interpolation.HERMITE)
    
    angle = 1.0e-3
    z0 = 20-3.4
    
    start_pos = np.array([0.0, z0])
    start_vel = T.velocity_vec(2000, angle, -1)
     
    st = time.time()
    _, pos_hermite = tracer_hermite(start_pos, start_vel)
    print(f'Trace took {(time.time()-st)*1000:.1f} ms')
     
    correct = 1.46214076e-02 # Determined by a accurate, naive trace
    int_hermite = T.plane_intersection(pos_hermite, z0)

    print(f'Calculated intersection: {int_hermite[0]:.4e} mm (correct: {correct:.4e} mm)')
     
    return exc.get_number_of_active_vertices(), abs(int_hermite[0]/correct - 1)


util.parser.description = '''
Consider the accuracy of Hermite interpolation by comparing a trace with the naive BEM tracing method
(iterating over all line elements for every field evaluation). The geometry is the micro mirror taken from:

H. Dohi, P. Kruit. Design for an aberration corrected scanning electron microscope using
miniature electron mirrors. 2018.
'''
util.parse_validation_args(create_geometry, compute_error, mirror='brown', lens='blue', ground='green',
    N={'radial': [100, 150, 300, 500]})


