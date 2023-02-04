from scipy.interpolate import *
import matplotlib.pyplot as plt
import time
import numpy as np

import traceon.geometry as G
import traceon.tracing as T
import traceon.aberrations as A
import traceon.excitation as E
import traceon.solver as solver

import util

# Correct values according to
# D. Preikszas and H Ros.e Correction properties of electron mirrors. 1997.
C30 = -0.61629*5 # Radius used is 5
C50 = -169.63*5 
C11 = 0.187461*5
C31 = 8.669*5
C12 = -0.0777*5

correct = [(3, 0, C30), (5, 0, C50), (1, 1, C11), (3, 1, C31), (1, 2, C12)]

def create_geometry(N):
    return G.create_preikszas_mirror(N)

def compute_error(N):
    excitation = E.Excitation(create_geometry(N))
    excitation.add_voltage(mirror=-250, corrector=1000)
      
    solution, f = solver.field_function_derivs(excitation, recompute=True)
    
    z = np.linspace(-0.1, -15)
    pot = [solver.potential_at_point(np.array([0.0, z_]), solution) for z_ in z]
    
    tracer = T.PlaneTracer(f, -38.688)
    
    C, dE, angles, intersections = A.compute_coefficients(tracer, 1000, dr=0.25)
    
    print('-'*20 + ' Accuracy')
    for i, j, c in correct:
        print(f'C{i}{j} = {C[i,j]:.4e} (correct: {1000*c:.4e}) \t~ {abs(C[i,j])/abs(c*1000) - 1:+.1e}')
    print('-'*20)
    
    return excitation.get_number_of_active_lines(), abs(C[3,0]/(C30*1000) - 1)

util.parser.description = '''
Calculate the aberration coefficients of a diode mirror. The accuracy plotted is determined from the 
spherical aberration coefficient, but the accuracy is printed for all coefficients. See paper:

Correction properties of electron mirrors. D. Preikszas and H. Rose. 1997.
'''
util.parse_validation_args(create_geometry, compute_error, mirror='blue', corrector='green')

