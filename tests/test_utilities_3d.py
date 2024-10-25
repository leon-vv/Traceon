import unittest

from scipy.constants import mu_0
import numpy as np

import traceon.backend as B


class TestUtilities3D(unittest.TestCase):
    
    def test_combine_elec_magnetic(self):
        
        for i in range(20):
            vel, elec, mag, current = np.random.rand(4, 3)
            result = B.combine_elec_magnetic_field(vel, elec, mag, current)
            assert np.allclose(result, elec + mu_0*np.cross(vel, mag + current))
