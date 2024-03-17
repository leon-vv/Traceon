import unittest
from math import *

from traceon.geometry import *


class PathTests(unittest.TestCase):
    def test_from_irregular_function(self):
        f = lambda x: [x, x**(3/2), 0.]
        y = Path.from_irregular_function(f)
        assert np.isclose(y.path_length, 1.43970987337155), y.path_length
        
        # Half the path length is reached at the special point x=0.5662942656295281
        a = 0.5662942656295281
        yhalf = Path.from_irregular_function(lambda x: [a*x, (a*x)**(3/2), 0.])
        assert np.isclose(yhalf.path_length, y.path_length/2)

        assert np.allclose(f(0.), y(0.))
        assert np.allclose(f(1.), y(y.path_length))
        assert np.allclose(f(a), y(0.5*y.path_length))
        assert np.allclose(yhalf.final_point(), y.middle_point())
        assert np.allclose(f(1.), y.final_point())
    
    def test_irregular_path_length(self):
        y = Path.from_irregular_function(lambda x: np.array([x, x**2, 0.]))
        assert np.isclose(y.path_length, 1.478942857544597), y.path_length
        y = Path.from_irregular_function(lambda x: np.array([5*x, (5*x)**2, 0.]))
        assert np.isclose(y.path_length, 25.87424479037671), y.path_length
        y = Path.from_irregular_function(lambda x: np.array([0., 5*x, (5*x)**2]))
        assert np.isclose(y.path_length, 25.87424479037671), y.path_length
        y = Path.from_irregular_function(lambda x: np.array([(5*x)**2, 0., 5*x]))

    def test_move(self):
        y = Path.from_irregular_function(lambda x: np.array([x, x, 0.]))
        y = y.move(dz=1.)
        assert np.allclose(y(sqrt(2)), [1., 1., 1.])
    
    def test_rotate(self):
        y = Path.from_irregular_function(lambda x: np.array([x, x, 0.]))
        y = y.rotate(Rz=pi/2)
        assert np.allclose(y(sqrt(0)), [0., 0., 0.])
        assert np.allclose(y(sqrt(2)), [-1., 1., 0.])
        y = y.rotate(Rz=pi/2)
        assert np.allclose(y(sqrt(0)), [0., 0., 0.])
        assert np.allclose(y(sqrt(2)), [-1., -1., 0.])
        y = y.rotate(Ry=pi/2)
        assert np.allclose(y(sqrt(0)), [0., 0., 0.])
        assert np.allclose(y(sqrt(2)), [0., -1., 1.])
        y = y.rotate(Rx=-pi/2)
        assert np.allclose(y(sqrt(0)), [0., 0., 0.])
        assert np.allclose(y(sqrt(2)), [0., 1., 1.])
     
    def test_rotate_around_point(self):
        origin = np.array([2., 1., -1.])
        y = Path.from_irregular_function(lambda x: origin + np.array([x, x, 0.]))
        y = y.rotate(Rz=pi/2, origin=origin)
        assert np.allclose(y(sqrt(0)), origin)
        assert np.allclose(y(sqrt(2)), origin + np.array([-1., 1., 0.]))
        y = y.rotate(Rz=pi/2, origin=origin)
        assert np.allclose(y(sqrt(0)), origin)
        assert np.allclose(y(sqrt(2)), origin + np.array([-1., -1., 0.]))
        y = y.rotate(Ry=pi/2, origin=origin)
        assert np.allclose(y(sqrt(0)), origin)
        assert np.allclose(y(sqrt(2)), origin + np.array([0., -1., 1.]))
        y = y.rotate(Rx=-pi/2, origin=origin)
        assert np.allclose(y(sqrt(0)), origin)
        assert np.allclose(y(sqrt(2)), origin + np.array([0., 1., 1.]))
    
    def test_discretize_path(self):
        path_length = 10 
        breakpoints = [3.33, 5., 9.]
         
        u = discretize_path(path_length, breakpoints, 1.)

        assert 0. in u
        assert 10 in u
        
        for b in breakpoints:
            assert b in u

      
class SurfaceTests(unittest.TestCase):
    
    def test_spanned_by_paths(self):
        y1 = Path.from_irregular_function(lambda x: [x, -1. - x, x]) 
        y2 = Path.from_irregular_function(lambda x: [x, 1. + x, x]) 
        surf = Surface.spanned_by_paths(y1, y2)

        p1 = surf.path_length1
        p2 = surf.path_length2
         
        assert np.allclose(surf(0., 0.), [0., -1., 0.])
        assert np.allclose(surf(0., p2), [0., 1., 0.])
        assert np.allclose(surf(p1, 0.), [1., -2., 1.])
        assert np.allclose(surf(p1, p2), [1., 2., 1.])








