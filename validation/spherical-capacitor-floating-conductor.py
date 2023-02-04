import time
import argparse

import numpy as np
from pygmsh import *
import matplotlib.pyplot as plt

import traceon.geometry as G
import traceon.excitation as E
import traceon.solver as S
import traceon.plotting as P

import util

r1 = 0.5
r2 = 1.0

r3 = 0.6
r4 = 0.9

def create_geometry(N):
    
    with occ.Geometry() as geom:
        center = geom.add_point([0.0, 0.0], 1/N)
        def add_shell(r, reorient=False):
            points = [[0,-r], [r, 0], [0,r]]
            p = [geom.add_point(p, 1/N) for p in points]

            if not reorient:
                return [geom.add_circle_arc(p[0], center, p[1]), geom.add_circle_arc(p[1], center, p[2])]
            else:
                return [geom.add_circle_arc(p[2], center, p[1]), geom.add_circle_arc(p[1], center, p[0])]
        
        l1 = add_shell(r1)
        d1 = add_shell(r3, reorient=True)
        d2 = add_shell(r4)
        l2 = add_shell(r2)
        
        geom.add_physical(l1, 'inner')
        geom.add_physical(l2, 'outer')
        geom.add_physical([*d1, *d2], 'floating')
        
        mesh = geom.generate_mesh(dim=1)
        geom = G.Geometry(mesh, N)

    return geom

def compute_error(N):

    geom = create_geometry(N)

    exc = E.Excitation(geom)
    exc.add_voltage(inner=1)
    exc.add_voltage(outer=0)
    exc.add_floating_conductor(floating=0)

    solution = S.solve_bem(exc)
    _, line_points, charges, _ = solution
    # Bit of hack, figure out a cleaner way to pass this around
    lines, names = exc.get_active_lines()

    field = S.field_at_point(np.array([(r3+r4)/2, 0.0]), solution)
    print('Electric field inside conductor: Er=%.2e, Ez=%.2e' % (field[0], field[1]))
     
    # Field should be zero
    return exc.get_number_of_active_lines(), abs(field[0])
     

util.parser.description = '''Compute the field of two concentric spheres with a layer of floating (voltage not fixed) neutrally charged conductor in between.
The accuracy of the solution is determined by considering whether Er=0, as the field inside the floating conductor should be zero.'''
util.parse_validation_args(create_geometry, compute_error, inner='blue', outer='darkblue', floating='green')

