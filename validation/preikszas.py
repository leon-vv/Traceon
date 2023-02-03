from scipy.interpolate import *
import matplotlib.pyplot as plt
import time
import numpy as np

import traceon.geometry as G
import traceon.tracing as T
import traceon.aberrations as A
import traceon.excitation as E
import traceon.solver as solver

# Correct values according to
# D. Preikszas and H Ros.e Correction properties of electron mirrors. 1997.
C30 = -0.61629*5 # Radius used is 5
C50 = -169.63*5  # Error in paper?
C11 = 0.187461*5
C31 = 8.669*5
C12 = -0.0777*5

correct = [(3, 0, C30), (5, 0, C50), (1, 1, C11), (3, 1, C31), (1, 2, C12)]

print('Correct values: ')
for i, j, c in correct:
    print(f'C{i}{j} = {c:+.4e}')

for n in [250, 500, 1000, 3000, 5000]:
    
    geom = G.create_preikszas_mirror(n)
    excitation = E.Excitation(geom)
    excitation.add_voltage(mirror=-250, corrector=1000)
    print('Computing trajectories...')
    
    solution, f = solver.field_function_derivs(excitation, recompute=True)
    
    z = np.linspace(-0.1, -15)
    pot = [solver.potential_at_point(np.array([0.0, z_]), solution) for z_ in z]
    
    #solver.benchmark_field_function(f)
    
    tracer = T.PlaneTracer(f, -38.688)

    #tracer._benchmark()

    C, dE, angles, intersections = A.compute_coefficients(tracer, 1000, dr=0.25)
    print(C)

    print('Relative accuracy: ')
    for i, j, c in correct:
        print(f'C{i}{j} ~ {abs(C[i,j])/abs(c*1000) - 1:+.3e}')
     
    #plt.figure()
    #plt.scatter(dE, intersections[:, 0])
    #plt.scatter(dE, C.intersection(dE, angles), marker='+')
    #plt.xlabel('Relative energy deviation')
    #plt.ylabel('z intersection')
    #plt.show()

