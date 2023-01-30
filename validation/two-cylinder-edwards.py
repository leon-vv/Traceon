import time
import sys

from scipy.interpolate import *
import matplotlib.pyplot as plt
import numpy as np

import traceon.geometry as G
import traceon.excitation as E
import traceon.solver as solver
import traceon.plotting as P

times = []
accuracies = []

# Compile
exc = E.Excitation(G.create_two_cylinder_lens(N=5))
exc.add_voltage(v1=0, v2=10)
solution = solver.solve_bem(exc)
solver.potential_at_point(np.array([0.0, 0.0]), solution)

def gap_voltage(x, y, _):
    return (y-9.9)/0.2 * 10

for n in np.linspace(1000, 4000, 10).astype(np.int32):
    st = time.time()
    print('Creating mesh')
    geom = G.create_two_cylinder_lens(N=n)
    
    exc = E.Excitation(geom)
    exc.add_voltage(v1=0, v2=10)#, gap=gap_voltage)
     
    solution = solver.solve_bem(exc)
    times.append(time.time()-st)
    
    edwards = np.array([5.0, 2.5966375108359858, 1.1195606398479115, .4448739946832647, .1720028130382, .065954697686])

    accs = []
        
    for i, z in enumerate(2*np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])):
        point = np.array([0.0, 10.0 - z])
        pot = solver.potential_at_point(point, solution)
        
        accuracy = pot/edwards[i] - 1
        accs.append(accuracy)
        
        print(point, pot, f'Accuracy: {accuracy:.3e}')
     
    accuracies.append(np.mean(np.abs(accs)))
    print(f'Calculating took {(time.time()-st):.3f} s')

plt.figure()
plt.plot(times, accuracies)
plt.scatter(times, accuracies)
plt.yscale('log')
plt.ylabel('Mean relative error')
plt.xlabel('Compuation time (s)')
plt.ylim(1e-6, 1e-2)
plt.xlim(0, 10)
(W, NF) = solver.WIDTHS_FAR_AWAY, solver.N_FACTOR
plt.show()


