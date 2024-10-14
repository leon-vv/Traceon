import unittest
from math import pi, sqrt

import numpy as np

import traceon.backend as B

class TestRadial(unittest.TestCase):
    def test_charge_radial_vertical(self):
        vertices = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1/3, 0.0],
            [1.0, 2/3, 0.0]])
        
        correct = 2*pi
        approx = B.charge_radial(vertices, 1.0);

        assert np.isclose(correct, approx)
    
    def test_charge_radial_horizontal(self):
        vertices = np.array([
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]])
         
        correct = 15*pi
        approx = B.charge_radial(vertices, 1.0);
         
        assert np.isclose(correct, approx)
    
    def test_charge_radial_skewed(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1/3, 1/3, 0.0],
            [2/3, 2/3, 0.0]])
        
        correct = pi*sqrt(2)
        approx = B.charge_radial(vertices, 1.0);

        assert np.isclose(correct, approx)
