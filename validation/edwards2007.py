import matplotlib.pyplot as plt
import numpy as np
from pygmsh import *
import time

import traceon.geometry as G
import traceon.excitation as E
import traceon.plotting as P
import traceon.solver as S

import util


def create_geometry(N, symmetry, for_plot):
    """Create the geometry g5 (figure 2) from the following paper:
    D. Edwards. High precision electrostatic potential calculations for cylindrically
    symmetric lenses. 2007.
    """
    with occ.Geometry() as geom:
        points = [
            [0, 0],
            [0, 5],
            [12, 5],
            [12, 15],
            [0, 15],
            [0, 20],
            [20, 20],
            [20, 0]
        ]
        
        lcar = 20/N
        
        if symmetry == '3d':
            points = [geom.add_point([p[0], 0.0, p[1]], lcar) for p in points]
        else:
            points = [geom.add_point(p, lcar) for p in points]
        
        l1 = geom.add_line(points[1], points[2])
        l2 = geom.add_line(points[2], points[3])
        l3 = geom.add_line(points[3], points[4])
        
        l4 = geom.add_line(points[0], points[-1])
        l5 = geom.add_line(points[-3], points[-2])
        l6 = geom.add_line(points[-2], points[-1])
        
        if symmetry == '3d':
            inner = G.revolve_around_optical_axis(geom, [l1, l2, l3])
            boundary = G.revolve_around_optical_axis(geom, [l4, l5, l6], factor=0.6 if for_plot else 1.0)
            
            geom.add_physical(inner, 'inner')
            geom.add_physical(boundary, 'boundary')
            
            return G.Geometry(geom.generate_mesh(dim=2), N, ((0,20),(0,20),(0,20)), symmetry='3d')
        elif symmetry == 'radial':
            geom.add_physical([l1, l2, l3], 'inner')
            geom.add_physical([l4, l5, l6], 'boundary')
            
            return G.Geometry(geom.generate_mesh(dim=1), N, ((0,20), (0,20)), symmetry='radial')

def compute_error(geometry):
     
    excitation = E.Excitation(geometry)
    excitation.add_voltage(boundary=0, inner=10)

    Nlines = excitation.get_number_of_active_vertices()
      
    field = S.solve_bem(excitation)

    st = time.time()
    if excitation.geometry.symmetry == '3d':
        pot = field.potential_at_point(np.array([12, 0.0, 4]))
    else:
        pot = field.potential_at_point(np.array([12, 4]))
    print(f'Time for computing potential {(time.time()-st)*1000:.2f} ms')
    
    correct = 6.69099430708
    print('Potential: ', pot)
    print('Correct: ', correct)
    return Nlines, abs(pot/correct - 1)

util.parser.description = '''Compute the potential at point (12, 4) inside two coaxial cylinders. See paper:

High precision electrostatic potential calculations for cylindrically symmetric lenses. David Edwards. 2007.
'''

util.parse_validation_args(create_geometry, compute_error, boundary='blue', inner='orange',
    N={'3d': [2, 5, 8, 10, 12], 'radial': [50, 100, 300, 500, 700]})

