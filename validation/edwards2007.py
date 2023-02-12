import matplotlib.pyplot as plt
import numpy as np
from pygmsh import *

import traceon.geometry as G
import traceon.excitation as E
import traceon.plotting as P
import traceon.solver as S

import util

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
        
        lcar = 20/N
        poly = geom.add_polygon(points, lcar)
        
        for key, indices in [('inner', [1, 2, 3]), ('boundary', [5,6,7])]:
            geom.add_physical([poly.curves[i] for i in indices], key)
        
        return G.Geometry(geom.generate_mesh(dim=1), N)

def compute_error(N):
     
    excitation = E.Excitation(create_geometry(N))
    excitation.add_voltage(boundary=0, inner=10)

    Nlines = excitation.get_number_of_active_vertices()

    field = S.solve_bem(excitation)
    pot = field.potential_at_point(np.array([12, 4]))
    correct = 6.69099430708
    return Nlines, abs(pot/correct - 1)

util.parser.description = '''Compute the potential at point (12, 4) inside two coaxial cylinders. See paper:

High precision electrostatic potential calculations for cylindrically symmetric lenses. David Edwards. 2007.
'''

util.parse_validation_args(create_geometry, compute_error, boundary='blue', inner='green')

