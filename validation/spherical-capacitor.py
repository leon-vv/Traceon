from math import cos

import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.excitation as E
import traceon.plotting as P
import traceon.solver as S
import traceon.tracing as T

import util

angle = 0.05

def create_geometry(N):
    return G.create_spherical_capacitor(N)

def compute_error(N):
    geom = create_geometry(N)
    exc = E.Excitation(geom)
    exc.add_voltage(inner=5/3, outer=3/5)
    
    solution = S.solve_bem(exc)
    correct = -10/(2/cos(angle)**2 - 1)
    assert -12.5 <= correct <= 7.5 # Between spheres
     
    position = np.array([0.0, 10.0])
    vel = np.array([np.cos(angle), -np.sin(angle)])*0.5930969604919433

    field = S.field_function_bem(solution)
    pos = T.trace_particle(position, vel, field, 12.5, -12.5, 12.5, rmin=-0.1)
    
    r_final = T.axis_intersection(pos)
     
    print(f'Correct intersection: {correct:.8f}')
    print(f'Computed intersection: {r_final:.8f}')
     
    return exc.get_number_of_active_lines(), abs(r_final/correct - 1)

util.parser.description = '''Trace electrons through a spherical capacitor. After the electron traces an arc through the capacitor, its intersection
with the axis is compared with the exact values given in following paper (first benchmark test):

Comparison of FDM, FEM and BEM for electrostatic charged particle optics. D. Cubric , B. Lencova, F.H. Read, J. Zlamal. 1999.
'''

util.parse_validation_args(create_geometry, compute_error,
    N=[10, 50, 100, 200, 300, 400],
    inner='blue', outer='darkblue')
