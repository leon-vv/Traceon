import numpy as np
import math as m

import traceon.backend as backend

def f1(t1, t2, x, y):
    return x**2

def print_error(val, correct):
    print('Error: %.3e' % (val/correct - 1))

print_error(backend.line_integral(np.array([0.0, 0.0]), np.array([-5.0, 0.0]), np.array([5.0, 0.0]), f1), 250/3)
print_error(backend.line_integral(np.array([0.0, 0.0]), np.array([5.0, 0.0]), np.array([-5.0, 0.0]), f1), 250/3)
print_error(backend.line_integral(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0]), f1), 1.0)
print_error(backend.line_integral(np.array([0.0, 0.0]), np.array([2.0, 0.0]), np.array([2.0, 2.0]), f1), 8.0)

def f2(t1, t2, x, y):
    return 1.0

print_error(backend.line_integral(np.array([0.0, 0.0]), np.array([-2.0, -2.0]), np.array([2.0, 2.0]), f2), m.sqrt(32))
    
def f3(t1, t2, x, y):
    return x**2 + y**2

print_error(backend.line_integral(np.array([0.0, 0.0]), np.array([-2.0, -2.0]), np.array([2.0, 2.0]), f3), 8/3*m.sqrt(32))

def f4(t1, t2, x, y):
    return m.cos(x) + m.sin(y)

print_error(backend.line_integral(np.array([0.0, 0.0]), np.array([-3.0, -2.0]), np.array([2.0, 5.0]), f4), 0.94720865504815726694278422253005)
#-(m.sqrt(74)*(5*m.cos(5)-7*m.sin(3)-7*m.sin(2)-5*m.cos(2)))/35)









