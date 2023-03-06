import numpy as np
import math as m

import traceon.backend as backend


def print_error(val, correct):
    print('Error: %.3e' % (val/correct - 1))

def integrate(fun):
    return backend.triangle_integral(np.array([0., 0, 0]), np.array([0., 0, 0]), np.array([2., 0, 0]), np.array([0., 1, 0]), fun)

print_error(integrate(lambda *args: 1), 1)
print_error(integrate(lambda *args: 1/2), 1/2)
print_error(integrate(lambda *args: args[3]), 2/3) # Integrate x
print_error(integrate(lambda *args: args[3]*args[4]), 1/6) # Integrate y**22
