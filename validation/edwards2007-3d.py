import matplotlib.pyplot as plt
import numpy as np
from pygmsh import *

import traceon.geometry as G
import traceon.excitation as E
import traceon.plotting as P
import traceon.solver as S

import util

def revolve_around_optical(geom, elements, factor=1.0):
    revolved = []
    
    for e in elements:
        
        top = e
        for i in range(4):
            top, extruded, lateral = geom.revolve(top, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], factor*0.5*np.pi)
            revolved.append(extruded)
    
    return revolved

def create_geometry(N):
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
        
        lcar = 4* 20/N
        points = [geom.add_point([p[0], 0.0, p[1]], lcar) for p in points]
        
        l1 = geom.add_line(points[1], points[2])
        l2 = geom.add_line(points[2], points[3])
        l3 = geom.add_line(points[3], points[4])
        
        l4 = geom.add_line(points[0], points[-1])
        l5 = geom.add_line(points[-3], points[-2])
        l6 = geom.add_line(points[-2], points[-1])

        inner = revolve_around_optical(geom, [l1, l2, l3])
        boundary = revolve_around_optical(geom, [l4, l5, l6])
         
        geom.add_physical(inner, 'inner')
        geom.add_physical(boundary, 'boundary')
         
        return G.Geometry(geom.generate_mesh(dim=2), N, symmetry='3d')

def compute_error(N):
     
    excitation = E.Excitation(create_geometry(N))
    excitation.add_voltage(boundary=0, inner=10)

    Nlines = excitation.get_number_of_active_vertices()
     
    field = S.solve_bem(excitation)
    pot = field.potential_at_point(np.array([12, 0.0, 4]))
    correct = 6.69099430708
    print('Potential: ', pot)
    print('Correct: ', correct)
    return Nlines, abs(pot/correct - 1)

util.parser.description = '''Compute the potential at point (12, 4) inside two coaxial cylinders. See paper:

High precision electrostatic potential calculations for cylindrically symmetric lenses. David Edwards. 2007.
'''

util.parse_validation_args(create_geometry, compute_error, boundary='blue', inner='orange', N=[10, 20, 30, 40, 50])

