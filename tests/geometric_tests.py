import os.path as path

import unittest
from math import *

import voltrace as v
from voltrace.geometry import *

class MeshTests(unittest.TestCase):

    def test_loading_mesh(self):
        input_file = path.join(path.dirname(__file__), 'world.stl')
        output_file = path.join(path.dirname(__file__), 'world_out.stl')

        m = v.Mesh.read_file(input_file)
        m.write_file(output_file)
        m2 = v.Mesh.read_file(output_file)
         
        assert np.allclose(m.points, m2.points)
        assert np.allclose(m.triangles, m2.triangles)

class PathTests(unittest.TestCase):
    def test_normalize(self):
        f = lambda x: [x, x**(3/2), 0.]
        y = Path(f,1.0).normalize()
        assert np.isclose(y.parameter_range, 1.43970987337155), y.parameter_range
        
        # Half the path length is reached at the special point x=0.5662942656295281
        a = 0.5662942656295281
        yhalf = Path(lambda x: [a*x, (a*x)**(3/2), 0.], 1.0).normalize()
        assert np.isclose(yhalf.parameter_range, y.parameter_range/2)

        assert np.allclose(f(0.), y(0.))
        assert np.allclose(f(1.), y(y.parameter_range))
        assert np.allclose(f(a), y(0.5*y.parameter_range))
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
     
    def test_normalize2(self):
        y = Path(lambda x: np.array([x, x**2, 0.]), 1.0).normalize()
        assert np.isclose(y.parameter_range, 1.478942857544597), y.parameter_range
        y = Path(lambda x: np.array([5*x, (5*x)**2, 0.]), 1.0).normalize()
        assert np.isclose(y.parameter_range, 25.87424479037671), y.parameter_range
        y = Path(lambda x: np.array([0., 5*x, (5*x)**2]), 1.0).normalize()
        assert np.isclose(y.parameter_range, 25.87424479037671), y.parameter_range
    
    def test_move(self):
        y = Path(lambda x: np.array([x, x, 0.]), 1.0).normalize()
        y = y.move(dz=1.)
        assert np.allclose(y(sqrt(2)), [1., 1., 1.])
    
    def test_rotate(self):
        y = Path(lambda x: np.array([x, x, 0.]), 1.0).normalize()
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
        y = Path(lambda x: origin + np.array([x, x, 0.]), 1.0).normalize()
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
    
    def test_rotate_around_axis(self):
        
        p = Path.line([0,0,0], [1,1,1])
        origin=[0,0,0]
        angle = pi/2
        p_rot_par = p.rotate_around_axis([1,1,1], angle, origin)
        assert np.allclose(p_rot_par.starting_point(), origin)
        assert np.allclose(p_rot_par.endpoint(), np.array([1,1,1]))

        p_rot_ort = p.rotate_around_axis([0,1,-1], angle, origin)
        assert np.allclose(p_rot_ort.starting_point(), origin)
        assert np.allclose(p_rot_ort.endpoint(), np.array([sqrt(2), -1/sqrt(2), -1/sqrt(2)]))

        # general example calculated by hand using Rodrigues' formula as follows:
        # Translate v to v' = v - origin. 
        # Rotate around [0,0,0] using v_rot' = v'cos(theta) + (k \cross v')sin(theta) + (k(k\dot v))(1-cos(theta))
        # Translate back to v_rot using v_rot = origin + v_rot'
        origin = [2,1,-1]
        p_rot_gen = p.rotate_around_axis([1,1,0], -pi/2, [2,1,-1])
        assert np.allclose(p_rot_gen.starting_point(), np.array([-0.207107,0.207107,-1.707107]))
        assert np.allclose(p_rot_gen.endpoint(), np.array([0.085786, 1.914214, -1.707107]))

        # 2pi rotation should map object to itself
        p_rot_2pi = p.rotate_around_axis([1,1,0], 2*pi, [2,1,-1])
        assert np.allclose(p.starting_point(), p_rot_2pi.starting_point())
        assert np.allclose(p.endpoint(), p_rot_2pi.endpoint())

    def test_discretize_path(self):
        parameter_range = 10 
        breakpoints = [3.33, 5., 9.]
        
        P = Path(lambda x: np.array([0., 0., 0.]), parameter_range, breakpoints)
        u = P._discretize(1., 1., 1)
        
        assert 0. in u
        assert 10 in u
        
        for b in breakpoints:
            assert b in u
    
    def test_arc(self):
        r = 2.
        p = Path.arc([0., 0., 0.], [r, 0., 0.], [0., r, 0.])
        assert np.isclose(p.parameter_range, 1/4 * 2*pi*r)
        assert np.allclose(p(1/8 * 2*pi*r), [r/sqrt(2), r/sqrt(2), 0.])
        
        center = np.array([1., -1, -1])
        p = Path.arc(center, center+np.array([r,0.,0.]), center+np.array([0.,r, 0.]))
        assert np.isclose(p.parameter_range, 1/4 * 2*pi*r)
        assert np.allclose(p(1/8 * 2*pi*r), center + np.array([r/sqrt(2), r/sqrt(2), 0.]))
        
        r = 3
        center = np.array([0, r, 0.])
        p = Path.arc(center, [0., 0., 0.], [0., r, r])
        assert np.isclose(p.parameter_range, 1/4* 2*pi*r)
        assert np.allclose(p(1/8 * 2*pi*r), center + np.array([0., -r/sqrt(2), r/sqrt(2)]))
          
        r = 3
        center = np.array([0, r, 0.])
        p = Path.arc(center, [0., 0., 0.], [0., r, r], reverse=True)
        assert np.isclose(p.parameter_range, 3/4* 2*pi*r)
        assert np.allclose(p(1/2 * 3/4*2*pi*r), center + np.array([0., +r/sqrt(2), -r/sqrt(2)]))
          
        r = 3
        center = np.array([0, r, 0.])
        p = Path.arc(center, [0., 0., 0.], [0., r, -r])
        assert np.isclose(p.parameter_range, 1/4* 2*pi*r)
        assert np.allclose(p(1/2 * 1/4*2*pi*r), center + np.array([0., -r/sqrt(2), -r/sqrt(2)]))
        
        r = 3
        center = np.array([0, r, 0.])
        p = Path.arc(center, [0., 0., 0.], [0., r, -r], reverse=True)
        assert np.isclose(p.parameter_range, 3/4* 2*pi*r)
        assert np.allclose(p(1/2 * 3/4*2*pi*r), center + np.array([0., r/sqrt(2), r/sqrt(2)]))

    def test_polar_arc(self):
        r = 1.0
        angle = pi/2
        p = Path.polar_arc(r, angle, start=[0,0,1], plane_normal=[0,1,0], direction=[1,0,0])
        
        expected_length = r* angle
        self.assertTrue(isclose(p.parameter_range, expected_length, rel_tol=1e-7))
        
        start_point = p(0.)
        end_point = p(p.parameter_range)

        assert np.allclose(start_point, [0., 0., 1.], atol=1e-7)
        assert np.allclose(end_point, [1., 0., 0.], atol=1e-7)

    def test_extend_with_polar_arc(self):
        base_path = Path.line([0.,0.,0.], [1.,0.,0.])
        extended_path = base_path.extend_with_polar_arc(radius=1.0, angle=pi/2, plane_normal=[0,1,0])

        expected_length = 1.0 + (pi/2)
        assert np.isclose(extended_path.parameter_range, expected_length)
        
        end_point = extended_path(extended_path.parameter_range)
        assert np.allclose(end_point, [2.,0.,-1.], atol=1e-7) # should with right hand rule around y

    def test_velocity_vector(self):
        circle = Path.circle_xy(x0=0.0, y0=0., radius=1., angle=2*pi)

        t_start = circle.velocity_vector(0.)
        t_start_norm = t_start / np.linalg.norm(t_start)
        assert np.allclose(t_start_norm, [0.,1.,0.], atol=1e-7)

        half_length = circle.parameter_range/2
        t_half = circle.velocity_vector(half_length)
        t_half_norm = t_half / np.linalg.norm(t_half)
        assert np.allclose(t_half_norm, [0.,-1.,0.], atol=1e-7)
        
class SurfaceTests(unittest.TestCase):
    
    def test_spanned_by_paths(self):
        y1 = Path(lambda x: [x, -1. - x, x], 1.0)
        y2 = Path(lambda x: [x, 1. + x, x], 1.0) 
        surf = Surface.spanned_by_paths(y1, y2)

        p1 = surf.parameter_range1
        p2 = surf.parameter_range2
         
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
            .extend_with_line(points[2]).extend_with_line(points[3]).revolve_z()
        
        inner.name = 'test'
        mesh = inner.mesh(mesh_size=10/2)
          
        for i, t in enumerate(mesh.triangles):
            p1, p2, p3 = mesh.points[t]
            assert not np.all(p1 == p2), (i, t, p1, p2, p3)
            assert not np.all(p2 == p3)
            assert not np.all(p3 == p1)
         
        assert len(mesh.physical_to_triangles['test']) == len(mesh.triangles)

    def test_interpolate(self):
        x = np.linspace(0, 10)
        points = [[x_, x_, 0.] for x_ in x]
        derivs = [[1., 1., 0.] for x_ in x]

        interp = Path.interpolate(x, points)
        
        for x_, p_ in zip(x, points):
            assert np.allclose(interp(x_), p_)
        
        interp_derivs = Path.interpolate(x, points, derivs)
        
        for x_, p_ in zip(x, points):
            assert np.allclose(interp_derivs(x_), p_)



