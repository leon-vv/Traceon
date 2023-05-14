import time, math

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pygmsh import *

import traceon.geometry as G
import traceon.excitation as E
import traceon.tracing as T
import traceon.solver as S

import util

def create_geometry(MSF, symmetry):
    
    with G.Geometry(symmetry, size_from_distance=True, zmin=0.1, zmax=2) as geom:
        
        points = [ [0, -1], [2, -1], [2, 1], [0.3, 1] ]
        if symmetry == G.Symmetry.THREE_D:
            points = [ [p[0], 0.0, p[1]] for p in points ]
        
        points = [geom.add_point(p) for p in points]
        
        ground_lines = [geom.add_line(p1, p2) for p1, p2 in zip(points, points[1:])]
            
        if symmetry == G.Symmetry.THREE_D:
            revolved = G.revolve_around_optical_axis(geom, ground_lines)
            geom.add_physical(revolved, 'ground')
        else:
            geom.add_physical(ground_lines, 'ground')
        
        points = [ [0.0, 0.0], [1, 0] ]
        if symmetry == G.Symmetry.THREE_D:
            points = [ [p[0], 0.0, p[1]] for p in points ]
          
        points = [geom.add_point(p) for p in points]
         
        mirror_line = geom.add_line(points[0], points[1])

        if symmetry == G.Symmetry.THREE_D:
            revolved = G.revolve_around_optical_axis(geom, mirror_line)
            geom.add_physical(revolved, 'mirror')
        else:
            geom.add_physical(mirror_line, 'mirror')
        
        geom.set_mesh_size_factor(MSF)
        return geom.generate_mesh()

def compute_error(geom):
    excitation = E.Excitation(geom)
    excitation.add_voltage(mirror=-110, ground=0.0)

    field = S.solve_bem(excitation)

    _3d = geom.symmetry == G.Symmetry.THREE_D

    bounds = ((-0.22, 0.22), (0.02, 11))

    if _3d:
        bounds = ((-0.22, 0.22), (-0.22, 0.22), (0.02, 11))
     
    axial_field = field.axial_derivative_interpolation(0.02, 4)
    tracer = T.Tracer(axial_field, bounds)
    
    pos = np.array([0.0, 10.0]) if not _3d else np.array([0.0, 0.0, 10.0])
    vel = T.velocity_vec_xz_plane(100, 1e-3, three_dimensional=_3d)
     
    st = time.time()
    _, pos = tracer(pos, vel)
    print(f'Trace took {(time.time()-st)*1000:.0f} ms')
    
    p = T.xy_plane_intersection(pos, 10)
     
    #correct = 0.16338325
    correct = 1.6327355811e-01
    calculated = np.linalg.norm(p[:2] if _3d else p[:1])
     
    print(f'Computed intersection: {calculated:.10e} (correct: {correct:.5e})')
    return excitation, calculated/correct - 1

util.parser.description = '''   '''
util.parse_validation_args(create_geometry, compute_error, mirror='brown', lens='blue', ground='green',
    MSF={'radial': [200, 300, 400, 500], '3d': [200, 250, 500, 1000]})


