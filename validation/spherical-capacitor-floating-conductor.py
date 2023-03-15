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

def create_geometry(MSF, symmetry, for_plot):
     
    with G.Geometry(symmetry) as geom:
        center = geom.add_point([0.0, 0.0])
        
        def add_shell(r, reorient=False, factor=1.0):
            
            if symmetry == G.Symmetry.RADIAL:
                points = [[0,-r], [r, 0], [0,r]]
            elif symmetry == G.Symmetry.THREE_D:
                points = [[0,0,-r], [r,0, 0], [0,0,r]]
            
            p = [geom.add_point(p) for p in points]

            if not reorient:
                arcs = [geom.add_circle_arc(p[0], center, p[1]), geom.add_circle_arc(p[1], center, p[2])]
            else:
                arcs = [geom.add_circle_arc(p[2], center, p[1]), geom.add_circle_arc(p[1], center, p[0])]

            if symmetry == G.Symmetry.THREE_D:
                return G.revolve_around_optical_axis(geom, arcs, factor=factor)
            else:
                return arcs
        
        l1 = add_shell(r1)
        d1 = add_shell(r3, reorient=True, factor=0.7 if for_plot else 1.0)
        d2 = add_shell(r4, factor=0.5 if for_plot else 1.0)
        l2 = add_shell(r2, factor=0.3 if for_plot else 1.0)
        
        geom.add_physical(l1, 'inner')
        geom.add_physical(l2, 'outer')
        geom.add_physical([*d1, *d2], 'floating')
        
        geom.set_mesh_size_factor(MSF)
        return geom.generate_mesh()

def compute_error(geom):
    exc = E.Excitation(geom)
    exc.add_voltage(inner=1)
    exc.add_voltage(outer=0)
    exc.add_floating_conductor(floating=0)

    field = S.solve_bem(exc)
    print('Floating voltages: ', field.floating_voltages)
    
    if geom.symmetry == G.Symmetry.RADIAL:
        point = np.array([(r3+r4)/2, 0.0])
        field_val = field.field_at_point(point)
        print('Electric field inside conductor: Er=%.2e, Ez=%.2e' % (field_val[0], field_val[1]))
    else:
        point = np.array([(r3+r4)/2, 0.0, 0.0])
        field_val = field.field_at_point(point)
        print('Electric field inside conductor: Ex=%.2e, Ey=%.2e, Ez=%.2e' % (field_val[0], field_val[1], field_val[2]))
    
    print('Potential inside conductor: ', field.potential_at_point(point))
     
    # Field should be zero
    return exc.get_number_of_active_elements(), field.potential_at_point(point)/0.25 - 1
     

util.parser.description = '''Compute the field of two concentric spheres with a layer of floating (voltage not fixed) neutrally charged conductor in between.
The accuracy of the solution is determined by considering whether Er=0, as the field inside the floating conductor should be zero.'''
util.parse_validation_args(create_geometry, compute_error, inner='blue', outer='darkblue', floating='green')

