import unittest

import numpy as np
from scipy.special import ellipe, ellipk, ellipkm1

import traceon.backend as B

class TestElliptic(unittest.TestCase):
    
    def test_ellipk(self):
        x = np.linspace(-1, 1, 40)[1:-1]
        # ellipk is slightly inaccurate, but is not used in electrostatic
        # solver. Only ellipkm1 is used
        assert np.allclose(B.ellipk(x), ellipk(x), atol=0., rtol=1e-5)
    
    def test_ellipe(self):
        x = np.linspace(-1, 1, 40)[1:-1]
        assert np.allclose(B.ellipe(x), ellipe(x), atol=0., rtol=1e-12)
 
    def test_ellipkm1_big(self):
        x = np.linspace(0, 1)[1:-1]
        assert np.allclose(ellipkm1(x), B.ellipkm1(x), atol=0., rtol=1e-12)
 
    def test_ellipkm1_small_many(self):
        x = np.linspace(1, 100, 5)
        assert np.allclose(ellipkm1(10**(-x)), B.ellipkm1(10**(-x)), atol=0., rtol=1e-12)
    
    def test_ellipem1_small_many(self):
        x = np.linspace(1, 100, 5)
        assert np.allclose(ellipe(1 - 10**(-x)), B.ellipem1(10**(-x)), atol=0., rtol=1e-12)


