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

def create_geometry(N, symmetry, for_plot):
     
    with occ.Geometry() as geom:
        center = geom.add_point([0.0, 0.0], 1/N)
        
        def add_shell(r, reorient=False, factor=1.0):
            
            if symmetry == 'radial':
                points = [[0,-r], [r, 0], [0,r]]
            elif symmetry == '3d':
                points = [[0,0,-r], [r,0, 0], [0,0,r]]
            
            p = [geom.add_point(p, 1/N) for p in points]

            if not reorient:
                arcs = [geom.add_circle_arc(p[0], center, p[1]), geom.add_circle_arc(p[1], center, p[2])]
            else:
                arcs = [geom.add_circle_arc(p[2], center, p[1]), geom.add_circle_arc(p[1], center, p[0])]

            if symmetry == '3d':
                return G.revolve_around_optical_axis(geom, arcs, factor=factor)
            else:
                return arcs
        
        l1 = add_shell(r1)
        d1 = add_shell(r3, reorient=True, factor=0.7 if for_plot else 1.0)
        d2 = add_shell(r4, factor=0.5 if for_plot else 1.0)
        l2 = add_shell(r2, factor=0.3 if for_plot else 1.0)
        
        geom.add_physical(l1, 'inner')
        geom.add_physical(l2, 'outer')
        geom.add_physical([*d1, *d2], 'dielectric')
        
        mesh = geom.generate_mesh(dim=1 if symmetry != '3d' else 2)
        geom = G.Geometry(mesh, N, symmetry=symmetry)
    
    return geom


def compute_error(geom):
    exc = E.Excitation(geom)
    exc.add_voltage(inner=1)
    exc.add_voltage(outer=0)
    K=2
    exc.add_dielectric(dielectric=K)

    field = S.solve_bem(exc)
    x = np.linspace(0.55, 0.95)
    f = [field.field_at_point(np.array([x_, 0.0, 0.0]))[0] for x_ in x]
    
    plt.plot(x, f)
    plt.show()
     

    
    vertices = field.vertices
    charges = field.charges
     
    # Find the charges
    Q = {}
     
    for n, v in field.names.items():
        Q[n] = 0
        
        for vs, charge in zip(vertices[v], charges[v]):
            
            if geom.symmetry == 'radial':
                v1, v2 = vs
                length = np.linalg.norm(v1 - v2)
                middle = (v1 + v2)/2
                # Take into account surface area of entire ring
                Q[n] += charge * length*2*np.pi*middle[0]
            elif geom.symmetry == '3d':
                v1, v2, v3 = vs
                area = 1/2*np.linalg.norm(np.cross(v2-v1, v3-v1))
                Q[n] += charge * area
    
    expected = 4/( (1/r1 - 1/r3) + (1/r3 - 1/r4)/K + (1/r4 - 1/r2))
    capacitance = (abs(Q['outer']) + abs(Q['inner']))/2
    print('Capacitance found: %.4f' % capacitance)
    print('Capacitance expected: %.4f' % expected)
    error = abs(capacitance/expected - 1)
    return len(vertices), error

util.parser.description = '''Compute the capacitance of two concentric spheres with a layer of dielectric material in between.'''
util.parse_validation_args(create_geometry, compute_error, inner='blue', outer='darkblue', dielectric='green',
    N={'radial':[10, 50, 100, 300,500,700],'3d':[2,5,7,9, 11]})
