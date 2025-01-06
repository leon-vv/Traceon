import os.path as path

import unittest
from math import *

import traceon.mesher as M
from traceon.geometry import *
import traceon.excitation as E
import traceon.solver as S
from traceon.field import FieldRadialAxial

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

    def test_polar_arc(self):
        r = 1.0
        angle = pi/2
        p = Path.polar_arc(r, angle, start=[0,0,1], plane_normal=[0,1,0], direction=[1,0,0])
        
        expected_length = r* angle
        self.assertTrue(isclose(p.path_length, expected_length, rel_tol=1e-7))
        
        start_point = p(0.)
        end_point = p(p.path_length)

        assert np.allclose(start_point, [0., 0., 1.], atol=1e-7)
        assert np.allclose(end_point, [1., 0., 0.], atol=1e-7)

    def test_extend_with_polar_arc(self):
        base_path = Path.line([0.,0.,0.], [1.,0.,0.])
        extended_path = base_path.extend_with_polar_arc(radius=1.0, angle=pi/2, plane_normal=[0,1,0])

        expected_length = 1.0 + (pi/2)
        assert np.isclose(extended_path.path_length, expected_length)
        
        end_point = extended_path(extended_path.path_length)
        assert np.allclose(end_point, [2.,0.,-1.], atol=1e-7) # should with right hand rule around y

    def test_velocity_vector(self):
        circle = Path.circle_xy(x0=0.0, y0=0., radius=1., angle=2*pi)

        t_start = circle.velocity_vector(0.)
        t_start_norm = t_start / np.linalg.norm(t_start)
        assert np.allclose(t_start_norm, [0.,1.,0.], atol=1e-7)

        half_length = circle.path_length/2
        t_half = circle.velocity_vector(half_length)
        t_half_norm = t_half / np.linalg.norm(t_half)
        assert np.allclose(t_half_norm, [0.,-1.,0.], atol=1e-7)
        
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
            .extend_with_line(points[2]).extend_with_line(points[3]).revolve_z()
        
        inner.name = 'test'
        mesh = inner.mesh(mesh_size=10/2)
          
        for i, t in enumerate(mesh.triangles):
            p1, p2, p3 = mesh.points[t]
            assert not np.all(p1 == p2), (i, t, p1, p2, p3)
            assert not np.all(p2 == p3)
            assert not np.all(p3 == p1)
         
        assert len(mesh.physical_to_triangles['test']) == len(mesh.triangles)
        
class FieldTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # simple field; details are not important for geometric tests
        pos = Path.rectangle_xz(0.1,1,1, 1.5)
        neg = Path.rectangle_xz(0.1,1,-1.5, -1)
        neg.name='neg'
        pos.name='pos'

        mesh = (neg + pos).mesh(mesh_size=1)

        excitation = E.Excitation(mesh, E.Symmetry.RADIAL)
        excitation.add_voltage(neg=-1, pos=1)
        excitation.add_magnetostatic_potential(neg=-1, pos=1)
        cls.field = S.solve_direct(excitation)
        cls.field_axial = FieldRadialAxial(cls.field, -2, 2, 100)

    def test_map_points(self):
        # translation
        field_trans = self.field.move(dx=1)
        # origin should shift
        assert np.allclose(field_trans.origin, [1.,0.,0.]) 
        # basis vectors should remain invariant
        assert np.allclose(field_trans.basis, self.field.basis) 
        # in homogenous coords the tranformation matrix is [[R, t], [0,1]]
        assert np.allclose(
            np.linalg.inv(field_trans.inverse_transformation_matrix), np.array([[1,0,0,1], 
                                                                                [0,1,0,0],
                                                                                [0,0,1,0],
                                                                                [0,0,0,1]]))
        #rotation after translation
        field_trans_rot = field_trans.rotate(Ry=np.pi/2)
        # local origin [1,0,0] rotates as well in global coordinate system
        assert np.allclose(field_trans_rot.origin, [0,0,-1]) 
        # x-> -z, y-> y, z-> -x
        assert np.allclose(field_trans_rot.basis, np.array([[0,0,-1],[0,1,0],[1,0,0]])) 
        #T_total = T_rot @ T_trans
        assert np.allclose(
            np.linalg.inv(field_trans_rot.inverse_transformation_matrix), np.array([[0,0,-1,0], 
                                                                                    [0,1,0,0],
                                                                                    [1,0,0,-1],
                                                                                    [0,0,0,1]]))
   
    def test_map_points_to_local(self):
        field_trans = self.field.move(dx=1)
        point = np.array([1,1,1])
        point_trans = point + np.array([1,0,0]) # point translated over same distance as field
        #translated point should be original point in translated system
        assert np.allclose(field_trans.map_points_to_local(point_trans), point)

        # origin should locally always coincide with standard origin
        field_trans_rot = field_trans.rotate(Ry=np.pi/2)
        assert np.allclose(field_trans_rot.map_points_to_local(field_trans_rot.origin), self.field.origin)
        # NOTE: map_points_to_local does not work for basis as they are direction vectors not points
    
    def test_field_axial_coordinate_system(self):
        field_axial_trans_rot1 = self.field_axial.move(dz=1).rotate(Rz=np.pi)

        field_trans_rot = self.field.move(dz=1).rotate(Rz=np.pi)
        field_axial_trans_rot2= FieldRadialAxial(field_trans_rot, -2, 2, 100)
        
        # it should not matter wheter we interpolate or transform first
        # FieldAxial should inherit the coordinate system of its base field
        assert np.allclose(field_axial_trans_rot1.inverse_transformation_matrix, 
                           field_axial_trans_rot2.inverse_transformation_matrix)
    
    def test_potential_at_point(self):
        field_trans = self.field.move(dx=1)
        field_axial_trans = self.field_axial.move(dx=1)

        pg = np.array([1,1,1])
        pl = field_trans.map_points_to_local(pg)

        assert np.allclose(field_trans.electrostatic_potential_at_point(pg),
                        field_trans.electrostatic_potential_at_local_point(pl))
        
        assert np.allclose(field_trans.magnetostatic_potential_at_point(pg),
                        field_trans.magnetostatic_potential_at_local_point(pl))
        
        assert np.allclose(field_axial_trans.electrostatic_potential_at_point(pg),
                        field_axial_trans.electrostatic_potential_at_local_point(pl))
        
        assert np.allclose(field_axial_trans.magnetostatic_potential_at_point(pg),
                        field_axial_trans.magnetostatic_potential_at_local_point(pl))
        
    def test_field_at_point(self):
        field_trans = self.field.move(dx=1)
        field_axial_trans = self.field_axial.move(dx=1)

        pg = np.array([1,1,1])
        pl = field_trans.map_points_to_local(pg)

        assert np.allclose(field_trans.electrostatic_field_at_point(pg),
                        field_trans.electrostatic_field_at_local_point(pl))
        
        assert np.allclose(field_trans.magnetostatic_field_at_point(pg),
                        field_trans.magnetostatic_field_at_local_point(pl))
        
        assert np.allclose(field_axial_trans.electrostatic_field_at_point(pg),
                        field_axial_trans.electrostatic_field_at_local_point(pl))
        
        assert np.allclose(field_axial_trans.magnetostatic_field_at_point(pg),
                        field_axial_trans.magnetostatic_field_at_local_point(pl))
