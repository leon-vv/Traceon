import unittest

import numpy as np

import voltrace as v

class TestFocus(unittest.TestCase):
    
    def test_focus_2d(self):
        v1 = v.velocity_vec(10, [-1e-3, 0., -1])
        v2 = v.velocity_vec(10, [1e-3, 0., -1])

        trajectory1 = v.Path.interpolate([0., 3.], [ np.zeros(3), 3*v1 ], [ v1, v1 ])
        trajectory2 = v.Path.interpolate([0., 3.], [ np.zeros(3), 3*v2 ], [ v2, v2 ])

        (x, y, z) = v.focus_position([trajectory1, trajectory2])
        assert np.isclose(x, 0) and np.isclose(y, 0) and np.isclose(z, 0, atol=1e-7)

        trajectory1 = trajectory1.move(dx=1)
        trajectory2 = trajectory2.move(dx=1)
        
        (x, y, z) = v.focus_position([trajectory1, trajectory2])
        assert np.isclose(x, 1) and np.isclose(y, 0) and np.isclose(z, 0, atol=1e-7)
        
        trajectory1 = trajectory1.move(dy=1, dz=1)
        trajectory2 = trajectory2.move(dy=1, dz=1)
        
        (x, y, z) = v.focus_position([trajectory1, trajectory2])
        assert np.isclose(x, 1) and np.isclose(y, 1) and np.isclose(z, 1, atol=1e-7)
    
    def test_focus_3d(self):
        v1 = v.velocity_vec_spherical(1, 0, 0)
        v2 = v.velocity_vec_spherical(5, 1/30, 1/30)
        v3 = v.velocity_vec_spherical(10, 1/30, np.pi/2)

        trajectory1 = v.Path.interpolate([0., 3.], [ np.zeros(3), 3*v1 ], [ v1, v1 ])
        trajectory2 = v.Path.interpolate([0., 3.], [ np.zeros(3), 3*v2 ], [ v2, v2 ])
        trajectory3 = v.Path.interpolate([0., 3.], [ np.zeros(3), 3*v3 ], [ v3, v3 ])
         
        assert np.allclose(v.focus_position([trajectory1, trajectory2, trajectory3]), [0., 0., 0.], atol=3e-6)
