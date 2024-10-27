import unittest

import numpy as np
from scipy.integrate import quad

import traceon.backend as B

class TestUtilities2D(unittest.TestCase):
    def test_position_and_jacobian_radial(self):
        line = np.array([
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0]])

        _, pos = B.position_and_jacobian_radial(-1, *line)
        assert np.all(np.isclose(pos, line[0, :2]))
        _, pos = B.position_and_jacobian_radial(-1 + 2/3, *line)
        assert np.all(np.isclose(pos, line[1, :2]))
        _, pos = B.position_and_jacobian_radial(-1 + 4/3, *line)
        assert np.all(np.isclose(pos, line[2, :2]))
        _, pos = B.position_and_jacobian_radial(1, *line)
        assert np.all(np.isclose(pos, line[3, :2]))
    
    def test_position_and_jacobian_radial_length(self):
        line = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]])
        
        length = quad(lambda x: B.position_and_jacobian_radial(x, *line)[0], -1, 1)[0]
        assert np.isclose(length, 1.5)
        
        line = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0]])
        
        jac, pos = B.fill_jacobian_buffer_radial(np.array([line]))
        assert np.isclose(np.sum(jac[0]), 1.5)

        line = np.array([[ 2.1       , 0., -0.74974852],
            [ 2.1       , 0., -0.77751699],
            [ 2.1       , 0., -0.80528545],
            [ 2.1       , 0., -0.83305391]])
             
        length = quad(lambda x: B.position_and_jacobian_radial(x, *line)[0], -1, 1)[0]
        assert np.isclose(length, line[0, 2] - line[3, 2])

        normal = np.array([-1., 0., 0.])
        middle = np.mean(line, axis=0)

        def field_dot_normal_integrand(x):
            jac, pos = B.position_and_jacobian_radial(x, *line)
            
            field = np.array([
                -B.dr1_potential_radial_ring(middle[0], middle[2], pos[0], pos[1]),
                -B.dz1_potential_radial_ring(middle[0], middle[2], pos[0], pos[1]),
                0.0])

            return jac * np.dot(normal, field)
        
        field_dot_normal = quad(field_dot_normal_integrand, -1, 1, points=[-1., 0., 1.])[0]
        assert np.isclose(field_dot_normal, -0.01893954382812056)
