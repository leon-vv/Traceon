from math import cos

import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.plotting as P
import traceon.solver as S
import traceon.tracing as T

angle = -0.1
correct = -10/(2/cos(angle)**2 - 1)

assert -12.5 <= correct <= 7.5 # Between spheres

print(f'Correct: {correct:.5f}')

Ns = [100, 200, 400, 800, 1200]

line_elements = []
errors = []

for N in Ns:
    sc = G.create_spherical_capacitor(N)
    lines, charges = S.solve_bem(sc.mesh, inner=5/3, outer=3/5)
    
    position = np.array([0.0, 10.0])
    vel = np.array([np.cos(angle)*0.5930969604919433, -np.sin(angle)])

    field = S.field_function_bem(lines, charges)
    pos = T.trace_particle(position, vel, field, 12.5, -12.5, 12.5, rmin=-0.1)
    
    r_final = T.axis_intersection(pos)
     
    line_elements.append(lines.shape[0])
    errors.append(r_final/correct - 1)
    print(f'Intersection: {r_final:.5f}, Accuracy: {errors[-1]:.1e}')

plt.plot(line_elements, np.abs(errors))
plt.scatter(line_elements, np.abs(errors))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Number of line elements')
plt.ylabel('Relative error')
plt.show()

