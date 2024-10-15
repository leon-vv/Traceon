import unittest
from math import pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.constants import epsilon_0

import traceon.backend as B
from tests.test_radial_ring import potential_of_ring_arbitrary

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

    def test_field_radial(self):
        vertices = np.array([
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0 + 1/3, 1.0 + 1/3, 0.0],
            [1.0 + 2/3, 1.0 + 2/3, 0.0]])
        
        r0, z0 = 2.0, -2.5
        
        delta = 1e-5
        
        def Er(r, z):
            dVdr = (potential_of_ring_arbitrary(1.0, r0 + delta, z0, r, z)
                        - potential_of_ring_arbitrary(1.0,r0 - delta, z0, r, z))/(2*delta)
            return -dVdr
        
        def Ez(r, z):
            dVdz = (potential_of_ring_arbitrary(1.0, r0, z0 + delta, r, z)
                        - potential_of_ring_arbitrary(1.0, r0, z0 - delta, r, z))/(2*delta)
            return -dVdz
        
        length = sqrt(2)
        Er = quad(lambda x: Er(1.0 + x, 1.0 + x), 0.0, 1.0)[0] * length
        Ez = quad(lambda x: Ez(1.0 + x, 1.0 + x), 0.0, 1.0)[0] * length

        jac, pos = B.fill_jacobian_buffer_radial(np.array([vertices]))
        charges = np.ones(len(jac))
        assert np.allclose(B.field_radial(np.array([r0, z0]), charges, jac, pos)/epsilon_0, [Er, Ez], atol=0.0, rtol=1e-10)
    
        

        
    

