import matplotlib.pyplot as plt
import numpy as np

import traceon.backend as backend

# Test trace function with particle starting at the origin
# and initial speed 3 mm/ns in the z direction. The particle
# experiences a constant force in the x direction resulting in
# a horizontal accelaration of 3 mm/ns^2.
# The path is easily calculated analytically (see below).
# It can be seen that for this problem the trace particle function
# finds the path extremely accurately (down to floating point machine epsilon).

def field(*_):
    EM = -0.1758820022723908 # e/m units ns and mm
    acceleration_x = 3 # mm/ns
    return np.array([acceleration_x/EM, 0.0, 0.0])

bounds = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, np.sqrt(12)+1))
times, positions = backend.trace_particle(np.zeros( (3,) ), np.array([0., 0., 3.]), field, bounds, 1e-10)

correct_x = 3/2*times**2
correct_z = 3*times

plt.plot(correct_x, correct_z, label='Exact path')
plt.plot(positions[:, 0], positions[:, 2], linestyle='dashed', label='Traced')
plt.xlabel('x (mm)')
plt.ylabel('z (mm)')
plt.legend()
plt.figure()

plt.plot(times, positions[:, 0]/correct_x - 1, label='x error')
plt.plot(times, positions[:, 2]/correct_z - 1, label='z error')
plt.ylabel('Relative error')
plt.xlabel('Time (ns)')
plt.legend()
plt.yscale('log')

plt.show()


