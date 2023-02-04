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
        geom.add_physical([*d1, *d2], 'dielectric')
        
        mesh = geom.generate_mesh(dim=1)
        geom = G.Geometry(mesh, N)

    return geom

def compute_error(N):

    geom = create_geometry(N)

    exc = E.Excitation(geom)
    exc.add_voltage(inner=1)
    exc.add_voltage(outer=0)
    K=2
    exc.add_dielectric(dielectric=K)

    solution = S.solve_bem(exc)
    _, line_points, charges, _ = solution
    # Bit of hack, figure out a cleaner way to pass this around
    lines, names = exc.get_active_lines()
     
    # Find the charges
    Q = {}
     
    for n, v in names.items():
        Q[n] = 0
        
        for line, charge in zip(line_points[v], charges[v]):
            
            length = np.linalg.norm(line[1] - line[0])
            middle = (line[1] + line[0])/2
            
            # Take into account surface area of entire ring
            Q[n] += charge * length*2*np.pi*middle[0]

    expected = 4/( (1/r1 - 1/r3) + (1/r3 - 1/r4)/K + (1/r4 - 1/r2))
    capacitance = (abs(Q['outer']) + abs(Q['inner']))/2
    print('Capacitance found: %.4f' % capacitance)
    print('Capacitance expected: %.4f' % expected)
    error = abs(capacitance/expected - 1)
    return len(lines), error

util.parser.description = '''Compute the capacitance of two concentric spheres with a layer of dielectric material in between.'''
util.parse_validation_args(create_geometry, compute_error, inner='blue', outer='darkblue', dielectric='green')
