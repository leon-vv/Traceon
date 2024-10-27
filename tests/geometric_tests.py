import os.path as path

import unittest
from math import *

import traceon.mesher as M
from traceon.geometry import *
import traceon.plotting as P

class MeshTests(unittest.TestCase):

    def test_loading_mesh(self):
        input_file = path.join(path.dirname(__file__), 'world.stl')
        output_file = path.join(path.dirname(__file__), 'world_out.stl')

        m = M.Mesh.read_file(input_file)
        m.write_file(output_file)
        m2 = M.Mesh.read_file(output_file)
         
        assert np.allclose(m.points, m2.points)
        assert np.allclose(m.triangles, m2.triangles)

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
        assert np.allclose(yhalf.endpoint(), y.middle_point())
        assert np.allclose(f(1.), y.endpoint())

    def test_spline_through_points(self):
        points = [
            [0, 0, 0],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
            [0, 0, 2]]

        path = Path.spline_through_points(points)
        assert np.allclose(path.starting_point(), points[0])
        assert np.allclose(path.endpoint(), points[-1])
     
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
    
    def test_arc(self):
        r = 2.
        p = Path.arc([0., 0., 0.], [r, 0., 0.], [0., r, 0.])
        assert np.isclose(p.path_length, 1/4 * 2*pi*r)
        assert np.allclose(p(1/8 * 2*pi*r), [r/sqrt(2), r/sqrt(2), 0.])
        
        center = np.array([1., -1, -1])
        p = Path.arc(center, center+np.array([r,0.,0.]), center+np.array([0.,r, 0.]))
        assert np.isclose(p.path_length, 1/4 * 2*pi*r)
        assert np.allclose(p(1/8 * 2*pi*r), center + np.array([r/sqrt(2), r/sqrt(2), 0.]))
        
        r = 3
        center = np.array([0, r, 0.])
        p = Path.arc(center, [0., 0., 0.], [0., r, r])
        assert np.isclose(p.path_length, 1/4* 2*pi*r)
        assert np.allclose(p(1/8 * 2*pi*r), center + np.array([0., -r/sqrt(2), r/sqrt(2)]))
          
        r = 3
        center = np.array([0, r, 0.])
        p = Path.arc(center, [0., 0., 0.], [0., r, r], reverse=True)
        assert np.isclose(p.path_length, 3/4* 2*pi*r)
        assert np.allclose(p(1/2 * 3/4*2*pi*r), center + np.array([0., +r/sqrt(2), -r/sqrt(2)]))
          
        r = 3
        center = np.array([0, r, 0.])
        p = Path.arc(center, [0., 0., 0.], [0., r, -r])
        assert np.isclose(p.path_length, 1/4* 2*pi*r)
        assert np.allclose(p(1/2 * 1/4*2*pi*r), center + np.array([0., -r/sqrt(2), -r/sqrt(2)]))
        
        r = 3
        center = np.array([0, r, 0.])
        p = Path.arc(center, [0., 0., 0.], [0., r, -r], reverse=True)
        assert np.isclose(p.path_length, 3/4* 2*pi*r)
        assert np.allclose(p(1/2 * 3/4*2*pi*r), center + np.array([0., r/sqrt(2), r/sqrt(2)]))
         
         


      
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

    def test_non_degenerate_triangles(self):
        # Found bug where a triangle had two common
        # points, making it have zero area.
        points = [[0, 0, 5],
            [12, 0, 5],
            [12, 0, 15],
            [0, 0, 15]]
        
        inner = Path.line(points[0], points[1])\
            .line_to(points[2]).line_to(points[3]).revolve_z()
        
        inner.name = 'test'
        mesh = inner.mesh(mesh_size=10/2)
          
        for i, t in enumerate(mesh.triangles):
            p1, p2, p3 = mesh.points[t]
            assert not np.all(p1 == p2), (i, t, p1, p2, p3)
            assert not np.all(p2 == p3)
            assert not np.all(p3 == p1)
         
        assert len(mesh.physical_to_triangles['test']) == len(mesh.triangles)
        










