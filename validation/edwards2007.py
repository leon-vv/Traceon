import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.excitation as E
import traceon.plotting as P
import traceon.solver as S

correct = 6.69099430708

Ns = [100, 200, 400, 800, 1200]

line_elements = []
errors = []

for N in Ns:
    edwards = G.create_edwards2007(N)
    
    excitation = E.Excitation(edwards)
    excitation.add_voltage(boundary=0, inner=10)
    
    solution = S.solve_bem(excitation)
    pot = S.potential_at_point(np.array([12, 4]), solution)
    line_elements.append(solution[1].shape[0])
    errors.append(pot/correct - 1)
    print(f'Potential: {pot:.5f}, Accuracy: {errors[-1]:.1e}')

plt.plot(line_elements, np.abs(errors))
plt.scatter(line_elements, np.abs(errors))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Number of line elements')
plt.ylabel('Relative error')
plt.show()

