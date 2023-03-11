import time, math

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pygmsh import *

import traceon.geometry as G
import traceon.excitation as E
import traceon.tracing as T
import traceon.solver as S
import traceon.interpolation as interpolation
import traceon.radial_series_interpolation_3d as radial_3d

import util

def create_geometry(N, symmetry, for_plot):
    
    MESH_SIZE = 1/N
     
    bounds = ( (-0.22, 0.22), (0.02, 4) )

    if symmetry == '3d':
        bounds = ( (-0.22, 0.22), (-0.22, 0.22), (0.02, 4) )
        revolve_factor = 1.0
        
        if for_plot:
            revolve_factor = 0.6
    
    with occ.Geometry() as geom:
        
        points = [ [0, -1], [2, -1], [2, 1], [0.3, 1] ]
        if symmetry == '3d':
            points = [ [p[0], 0.0, p[1]] for p in points ]
        
        points = [geom.add_point(p, MESH_SIZE) for p in points]
        
        ground_lines = [geom.add_line(p1, p2) for p1, p2 in zip(points, points[1:])]
            
        if symmetry == '3d':
            revolved = G.revolve_around_optical_axis(geom, ground_lines, revolve_factor)
            geom.add_physical(revolved, 'ground')
        else:
            geom.add_physical(ground_lines, 'ground')
        
        points = [ [0.0, 0.0], [1, 0] ]
        if symmetry == '3d':
            points = [ [p[0], 0.0, p[1]] for p in points ]
          
        points = [geom.add_point(p, MESH_SIZE) for p in points]
         
        mirror_line = geom.add_line(points[0], points[1])

        if symmetry == '3d':
            revolved = G.revolve_around_optical_axis(geom, mirror_line, revolve_factor)
            geom.add_physical(revolved, 'mirror')
        else:
            geom.add_physical(mirror_line, 'mirror')
        
        return G.Geometry(geom.generate_mesh(dim=1 if symmetry != '3d' else 2), N, bounds, symmetry=symmetry)

def compute_error(geom):
    excitation = E.Excitation(geom)
    excitation.add_voltage(mirror=-110, ground=0.0)

    field = S.solve_bem(excitation)

    _3d = geom.symmetry == '3d'

    bounds = ((-0.22, 0.22), (0.02, 11))

    if _3d:
        bounds = ((-0.22, 0.22), (-0.22, 0.22), (0.02, 11))
     
    tracer = T.Tracer(field, bounds, T.Interpolation.AXIAL_DERIVS)
    
    pos = np.array([0.0, 10.0]) if not _3d else np.array([0.0, 0.0, 10.0])
    vel = T.velocity_vec_xz_plane(100, 1e-3, three_dimensional=_3d)
     
    st = time.time()
    _, pos = tracer(pos, vel)
    print(f'Trace took {(time.time()-st)*1000:.0f} ms')
    
    p = T.xy_plane_intersection(pos, 10)
     
    plt.plot(pos[:, 2 if _3d else 1], pos[:, 0])
    plt.show()
    correct = 0.16338325
    calculated = np.linalg.norm(p[:2] if _3d else p[:1])
     
    print(f'Computed intersection: {calculated:.4e} (correct: {correct:.4e})')
    return excitation.get_number_of_active_vertices(), calculated/correct - 1

util.parser.description = '''   '''
util.parse_validation_args(create_geometry, compute_error, mirror='brown', lens='blue', ground='green',
    N={'radial': [50, 100, 200, 300, 500], '3d':[1, 2, 4, 6]})


