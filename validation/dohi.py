import time

import numpy as np
import matplotlib.pyplot as plt

import traceon.geometry as G
import traceon.excitation as E
import traceon.tracing as T
import traceon.solver as S
import traceon.interpolation as interpolation

import util

def create_geometry(N, symmetry, for_plot):
    
    MESH_SIZE = 1/N

    revolve_factor = 0.0

    bounds = ((-0.03, 0.03), (0.1, 2.0))

    if symmetry == '3d':
        bounds = (bounds[0], (-0.03, 0.03), bounds[1])
        revolve_factor = 1.0 if not for_plot else 0.6
        rmax = 1.25
    else:
        rmax = 1.25
    
    with G.MEMSStack(bounds, z0=-0.3-0.15, revolve_factor=revolve_factor, rmax=rmax, margin_right=0.2, mesh_size=MESH_SIZE) as geom:
        
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
    exc.add_voltage(ground=0.0, mirror=-2139.1093, lens=-1200)
    
    field = S.solve_bem(exc)

    ## HERMITE validation
    #coeffs = field.get_hermite_interpolation_coeffs(sampling_factor=3)
    #x, zi = coeffs[0], coeffs[2]
    #x_ = x[x.size//2]
    
    #z = np.linspace(0.2, 1.5, 200)
    #Ex = [field(np.array([x_, 0.0, z_]))[0] for z_ in z]
     
    #Ex_inter = [interpolation.compute_hermite_field_3d(np.array([x_, 0.0, z_]), *coeffs)[0] for z_ in z]
    #plt.plot(z, Ex)
    #plt.plot(z, Ex_inter, linestyle='dashed')
    #for z_ in zi:
    #    plt.axvline(z_)
    #plt.show()
      
    bounds = ((-0.03, 0.03), (-0.03, 0.03), (0.05, 19.0)) if geom.symmetry == '3d' else ((-0.03, 0.03), (0.05, 19.0))
    tracer_hermite = T.Tracer(field, bounds, T.Interpolation.HERMITE)
    
    angle = 1.0e-3
    z0 = 20-3.4
    
    start_pos = np.array([0.0, 0.0, z0]) if geom.symmetry == '3d' else np.array([0.0, z0])
    start_vel = T.velocity_vec(2000, angle, -1, three_dimensional=geom.symmetry == '3d')
     
    print('Starting trace.')
    st = time.time()
    _, pos_hermite = tracer_hermite(start_pos, start_vel)
    print(f'Trace took {(time.time()-st)*1000:.1f} ms')
    
    plt.figure()
    idx = 2 if geom.symmetry == '3d' else 1
    plt.plot(pos_hermite[:, idx], pos_hermite[:, 0])
    plt.scatter(pos_hermite[:, idx], pos_hermite[:, 0])
    plt.show()
     
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
    N={'radial': [100, 150, 300, 500], '3d': [5, 10, 20]})


