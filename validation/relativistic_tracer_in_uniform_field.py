import matplotlib.pyplot as plt
import numpy as np
import scipy
import traceon.geometry as G
import traceon.solver as S
import traceon.excitation as E
import traceon.plotting as P
import traceon.tracing as T
import traceon.backend as B
from traceon.interpolation import FieldRadialAxial
from scipy.constants import m_e, e, mu_0, epsilon_0, c


#Initialize accaleration, constants, and initial speed
F = 1e9
EM = e/m_e
v_0 = 1.

#Uniform electric field in the x-direction
def acceleration(*_):
    return np.array([F, 0., 0.])

def field(*_):
    return acceleration() / EM

#trace
bounds = ((-2.0, 15e7), (-2.0, 2.0), (-2.0, 2.))
times_rel, positions_rel = B.trace_particle(np.zeros( (3,) ), np.array([v_0, 0., 0.]), EM, True, field, bounds, 1e-8)
times_classical, positions_classical = B.trace_particle(np.zeros( (3,) ), np.array([v_0, 0., 0.]), EM, False, field, bounds, 1e-8)

#Analytical solutions
correct_x_class = F/2*times_classical**2 + v_0*times_classical
correct_x_rel = c**2/F * (np.sqrt((F*times_rel + v_0)**2/c**2 + 1) - np.sqrt(1+(v_0)**2/c**2))


#Check whether tracers agree with analytical solution
assert np.allclose(positions_classical[:,0], correct_x_class)
assert np.allclose(positions_rel[:,0], correct_x_rel, atol =1e-8)

#Make a plot
plt.plot(times_rel, positions_rel[:,0], label = 'Relativistic')
plt.plot(times_classical, positions_classical[:,0], label = 'classical')
plt.plot(times_rel, times_rel*c, label = 'Light ray', linestyle = '--' )
plt.xlabel('t (seconds)')
plt.ylabel('x (meter)')
plt.title(f'Uniform force  1e9 m/s^2')
plt.legend()
plt.show()