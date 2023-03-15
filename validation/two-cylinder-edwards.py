import time
import sys

from scipy.interpolate import *
import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.excitation as E
import traceon.solver as solver
import traceon.plotting as P

import util


def create_geometry(MSF, symmetry, for_plot):
    assert symmetry == G.Symmetry.RADIAL, 'Only radial symmetry supported'
    
    S = 0.2
    R = 1.0
    wall_thickness = 1
    boundary_length = 20
    
    with G.Geometry(G.Symmetry.RADIAL) as geom:
        cylinder_length = (boundary_length - S)/2
        assert boundary_length == 2*cylinder_length + S

        assert wall_thickness == 0.0 or wall_thickness > 0.1
         
        points = [
            [0, 0],
            [R, 0],
            [R, cylinder_length],
            [R + wall_thickness, cylinder_length],
            [R + wall_thickness, cylinder_length + S],
            [R, cylinder_length + S],
            [R, boundary_length],
            [0, boundary_length]
        ]
            
        physicals = [('v1', [0, 1, 2]), ('v2', [4, 5, 6]), ('gap', [3])]
         
        poly = geom.add_polygon(points)
         
        for key, indices in physicals:
            lines = [poly.curves[idx] for idx in indices]
            geom.add_physical(lines, key)

        geom.set_mesh_size_factor(MSF)
        return geom.generate_mesh()




def gap_voltage(x, y, _):
    return (y-9.9)/0.2 * 10

def compute_error(geom):
    exc = E.Excitation(geom)
    exc.add_voltage(v1=0, v2=10, gap=gap_voltage)
     
    field = solver.solve_bem(exc)
    
    edwards = np.array([5.0, 2.5966375108359858, 1.1195606398479115, .4448739946832647, .1720028130382, .065954697686])
    z = 2*np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    point = np.array([0.0, 10.0 - z[1]])
    pot = field.potential_at_point(point)
     
    accuracy = abs(pot/edwards[1] - 1)
    return exc.get_number_of_active_vertices(), accuracy

util.parser.description = '''
Compute the potential inside a two cylinder lens.  The correct values for the potential are taken from the paper:\n
Accurate Potential Calculations For The Two Tube Electrostatic Lens Using A Multiregion FDM Method.  David Edwards, Jr. 2007.
'''

util.parse_validation_args(create_geometry, compute_error, v1='blue', v2='green', gap='orange')

