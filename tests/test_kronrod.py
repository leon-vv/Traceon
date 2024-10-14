from math import *

import unittest

import numpy as np

from scipy.integrate import quad

import traceon.backend as B

class TestKronrod(unittest.TestCase):
    def test_constant_function(self):
        f = lambda x: 1
        result = B.kronrod_adaptive(f, 0.0, 1.0)
        assert np.isclose(result, 1.0)

    def test_linear_function(self):
        f = lambda x: x
        result = B.kronrod_adaptive(f, 0.0, 1.0)
        assert np.isclose(result, 0.5)

    def test_quadratic_function(self):
        f = lambda x: x ** 2
        result = B.kronrod_adaptive(f, 0.0, 1.0)
        assert np.isclose(result, 1.0 / 3.0)

    def test_sine_function(self):
        f = lambda x: sin(x)
        result = B.kronrod_adaptive(f, 0.0, pi)
        assert np.isclose(result, 2.0)

    def test_difficult_exponential(self):
        f = lambda x: exp(x*x * cos(10*x))
        result = B.kronrod_adaptive(f, -1.5, 1.5)
        assert np.isclose(result, 4.097655169215941)
    
    def test_almost_singular_function(self):
        f = lambda x: 1/(x + 0.001);
        result = B.kronrod_adaptive(f, 0, 1)
        assert np.isclose(result, 6.90875477931522)
    
    



    

