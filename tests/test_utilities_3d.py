import unittest

from scipy.constants import mu_0
import numpy as np

import traceon.backend as B


class TestUtilities3D(unittest.TestCase):
    
    def test_normal_3d(self):
        tri = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0]])
        
        normal = B.normal_3d(tri)
        
        vec1 = tri[1] - tri[0]
        vec2 = tri[2] - tri[0]
         
        assert np.isclose(np.linalg.norm(normal), 1.0)
        assert np.isclose(np.dot(normal, vec1), 0.0)
        assert np.isclose(np.dot(normal, vec2), 0.0)

    def test_position_and_jacobian(self):
        tri = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0]])
        
        v0, v1, v2 = tri 
        
        vec1 = v1 - v0
        vec2 = v2 - v0
        
        alpha = 1/3
        beta = 1/4
         
        _, pos = B.position_and_jacobian_3d(alpha, beta, tri)
        
        assert np.allclose(pos, v0 + vec1*alpha + beta*vec2)
     
    def test_combine_elec_magnetic(self):
        
        for i in range(20):
            vel, elec, mag, current = np.random.rand(4, 3)
            result = B.combine_elec_magnetic_field(vel, elec, mag, current)
            assert np.allclose(result, elec + mu_0*np.cross(vel, mag + current))
