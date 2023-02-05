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


def create_geometry(N):
    return G.create_two_cylinder_lens(N)

def gap_voltage(x, y, _):
    return (y-9.9)/0.2 * 10

def compute_error(N):
    geom = create_geometry(N)
    exc = E.Excitation(geom)
    exc.add_voltage(v1=0, v2=10, gap=gap_voltage)
     
    field = solver.solve_bem(exc)
    
    edwards = np.array([5.0, 2.5966375108359858, 1.1195606398479115, .4448739946832647, .1720028130382, .065954697686])
    z = 2*np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    point = np.array([0.0, 10.0 - z[1]])
    pot = field.potential_at_point(point)
     
    accuracy = abs(pot/edwards[1] - 1)
    return exc.get_number_of_active_lines(), accuracy

util.parser.description = '''
Compute the potential inside a two cylinder lens.  The correct values for the potential are taken from the paper:\n
Accurate Potential Calculations For The Two Tube Electrostatic Lens Using A Multiregion FDM Method.  David Edwards, Jr. 2007.
'''

util.parse_validation_args(create_geometry, compute_error, v1='blue', v2='green', gap='orange',
    N=[10, 100, 500, 1000, 2000, 3000, 4000])

