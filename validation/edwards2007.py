import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.excitation as E
import traceon.plotting as P
import traceon.solver as S

import util

def create_geometry(N):
    return G.create_edwards2007(N)

def compute_error(N):
     
    excitation = E.Excitation(create_geometry(N))
    excitation.add_voltage(boundary=0, inner=10)

    Nlines = excitation.get_number_of_active_lines()

    solution = S.solve_bem(excitation)
    pot = S.potential_at_point(np.array([12, 4]), solution)
    correct = 6.69099430708
    return Nlines, abs(pot/correct - 1)

util.parse_validation_args(create_geometry, compute_error, boundary='blue', inner='green')

