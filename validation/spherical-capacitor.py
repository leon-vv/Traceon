from math import cos

import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.plotting as P
import traceon.solver as S
import traceon.tracing as T

Ns = [100, 200, 400, 800, 1200]

for N in Ns:
    sc = G.create_spherical_capacitor(N)
    solution = S.solve_bem(sc, inner=5/3, outer=3/5)
    
    for angle in [0.0, 0.05, 0.10]:
        print(f'Angle: {angle:.8f}')
        correct = -10/(2/cos(angle)**2 - 1)

        assert -12.5 <= correct <= 7.5 # Between spheres

        print(f'Correct: {correct:.8f}')
        
        line_elements = []
        errors = []
        
        position = np.array([0.0, 10.0])
        vel = np.array([np.cos(angle), -np.sin(angle)])*0.5930969604919433

        field = S.field_function_bem(solution)
        pos = T.trace_particle(position, vel, field, 12.5, -12.5, 12.5, rmin=-0.1)
        
        r_final = T.axis_intersection(pos)
        
        line_elements.append(solution[1].shape[0])
        errors.append(r_final/correct - 1)
        print(f'Intersection: {r_final:.8f}, Accuracy: {errors[-1]:.1e}')

'''
plt.plot(line_elements, np.abs(errors))
plt.scatter(line_elements, np.abs(errors))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Number of line elements')
plt.ylabel('Relative error')
plt.show()
'''


