import unittest
from math import pi, sqrt, atan2
import os.path as path

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.constants import epsilon_0, mu_0
from scipy.interpolate import CubicSpline

import traceon.geometry as G
import traceon.plotting as P
import traceon.solver as S
import traceon.excitation as E
import traceon.tracing as T
import traceon.backend as B
import traceon.logging as logging

logging.set_log_level(logging.LogLevel.SILENT)

class TestThreeD(unittest.TestCase):
    def test_revolved_rectangle(self):
        #Define surface
        THICKNESS = 1
        path = G.Path.line([0.0, 0.0, 0.0], [0.0, 0.0,THICKNESS])\
            .line_to([1., 0.0, THICKNESS])\
            .line_to([1., 0.0, 0.0])\
            .close()

        surf = path.revolve_z()
        surf.name = 'revolvedrectangle'

        #create mesh
        mesh = surf.mesh(mesh_size_factor=12)
        
        #add excitation
        excitation = E.Excitation(mesh, E.Symmetry.THREE_D)
        excitation.add_voltage(revolvedrectangle = -5.)

        #try solving
        field = S.solve_direct(excitation)

        z = np.array([0.25, 0.5, 0.75])
        axial_potential = [field.potential_at_point([0.0, 0.0, z_]) for z_ in [0.25, 0.5, 0.75]]

        assert np.allclose(axial_potential, -5, rtol=1e-4)

    def test_dohi_meshing(self):
        rmax = 1.0
        margin = 0.3
        extent = rmax-0.1

        t = 0.15 # thickness
        r = 0.075 # radius
        st = 0.5  # spacer thickness

        mirror = G.Path.aperture(0.15, r, extent, z=t/2)
        mirror.name = 'mirror'
        
        mirror_line = G.Path.line([0., 0., 0.], [r, 0., 0.])
        mirror_line.name = 'mirror'

        lens = G.Path.aperture(0.15, r, extent, z=t + st + t/2)
        lens.name = 'lens'
        
        ground = G.Path.aperture(0.15, r, extent, z=t+st+t+st+t/2)
        ground.name = 'ground'
    
        boundary = G.Path.line([0., 0., 1.75], [rmax, 0., 1.75]) \
            .line_to([rmax, 0., -0.3]).line_to([0., 0., -0.3])
        boundary.name = 'boundary'
        
        geom = mirror+mirror_line+lens+ground+boundary
        geom = geom.revolve_z()
        geom.mesh(mesh_size_factor=4)

    def test_simple_current_line_against_radial(self):
        rect = G.Path.rectangle_xz(1, 2, -1, 1)
        rect.name = 'rect'

        coil = G.Surface.disk_xz(3, 0, 0.05)
        coil.name = 'coil'

        mesh_size = 0.0125
        mesh = rect.mesh(mesh_size=mesh_size) + coil.mesh(mesh_size=mesh_size)

        e = E.Excitation(mesh, E.Symmetry.RADIAL)
        e.add_current(coil=2.5)
        e.add_magnetizable(rect=10)

        field = S.solve_direct(e)

        z = np.linspace(-10, 10, 50)
        pot_radial = [field.potential_at_point([0., 0., z_]) for z_ in z]

        rect = rect.revolve_z()
        coil = G.Path.circle_xy(0., 0., 3)
        coil.name = 'coil'

        mesh_size = 0.70
        mesh = rect.mesh(mesh_size=mesh_size) + coil.mesh(mesh_size=0.2)

        e = E.Excitation(mesh, E.Symmetry.THREE_D)
        e.add_current(coil=2.5)
        e.add_magnetizable(rect=10)

        field_three_d = S.solve_direct(e)
        pot_three_d = [field_three_d.potential_at_point([0., 0., z_]) for z_ in z]
        
        # Accuracy is better if mesh size is increased, but
        # makes tests too slow
        assert np.allclose(pot_three_d, pot_radial, atol=0.020)
    
    def test_magnetostatic_potential_against_radial_symmetric(self):
        rect = G.Path.rectangle_xz(1, 2, -1, 1)
        rect.name = 'rect'

        mesh_size = 0.02
        mesh = rect.mesh(mesh_size=mesh_size)
        
        e = E.Excitation(mesh, E.Symmetry.RADIAL)
        e.add_magnetostatic_potential(rect=10)

        field = S.solve_direct(e)

        z = np.linspace(-10, 10, 50)
        pot_radial = [field.potential_at_point([0.25, 0., z_]) for z_ in z]

        rect = rect.revolve_z()

        mesh_size = 0.70
        mesh = rect.mesh(mesh_size=mesh_size)
          
        e = E.Excitation(mesh, E.Symmetry.THREE_D)
        e.add_magnetostatic_potential(rect=10)

        field_three_d = S.solve_direct(e)
        pot_three_d = [field_three_d.potential_at_point([0.25, 0., z_]) for z_ in z]
        
        # Accuracy is better if mesh size is increased, but
        # makes tests too slow
        assert np.allclose(pot_three_d, pot_radial, rtol=8e-3)


class TestCurrentLoop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.radius = 5.
        cls.current = 2.5
        
        path = G.Path.circle_xy(0., 0., cls.radius)
        path.name = 'loop'

        mesh = path.mesh(mesh_size_factor=200)
        
        exc = E.Excitation(mesh, E.Symmetry.THREE_D)
        exc.add_current(loop=cls.current)

        solver = S.MagnetostaticSolver(exc)
        cls.field = solver.current_field
        
        # Now for axial
        small_circle = G.Surface.disk_xz(5., 0., 0.01)
        small_circle.name = 'loop'

        import traceon.plotting as P

        mesh = small_circle.mesh(mesh_size_factor=10)._to_higher_order_mesh()
        exc = E.Excitation(mesh, E.Symmetry.RADIAL)
        exc.add_current(loop=cls.current)
        cls.axial_field = S.MagnetostaticSolver(exc).current_field

    def test_field_on_axis(self):
        z = np.linspace(-20, 20, 25)
        # http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/curloo.html
        correct = 1/2 * self.current * self.radius**2 / (z**2 + self.radius**2)**(3/2)
        approx = np.array([self.field.current_field_at_point([0., 0., z_]) for z_ in z])

        assert np.allclose(approx[:, 0], 0.0)
        assert np.allclose(approx[:, 1], 0.0)
        assert np.allclose(correct, approx[:, 2], rtol=1e-4)
    
    def test_field_in_loop(self):
        def get_exact(x_):
            radius = self.radius
            f = lambda t: -self.current*radius**2/(4*np.pi*x_**3) * (x_/radius * np.cos(t) - 1)*(1 + radius**2/x_**2 - 2*radius/x_*np.cos(t))**(-3/2)
            return quad(f, 0, 2*np.pi)[0]

        x = np.concatenate( (np.linspace(0.25*self.radius, 0.75*self.radius, 10), np.linspace(1.25*self.radius, 2*self.radius, 10)) )
        correct = [get_exact(x_) for x_ in x]
         
        approx = np.array([self.field.current_field_at_point([x_, 0., 0.]) for x_ in x])
        
        assert np.allclose(approx[:, 0], 0.0)
        assert np.allclose(approx[:, 1], 0.0)
        assert np.allclose(correct, approx[:, 2], rtol=1e-4)

    def test_field_with_radial_symmetric(self):
        points = [
            [0., 0., 0.], # Bunch of arbitrary points
            [1., 0., 0.],
            [0., 0., 1.],
            [1., 1., 1.],
            [-1., 1., -1.],
            [2., 2., 2.],
            [2., 2., 0.],
            [4., 4., 3.],
            [-4., -4., -3.],
            [np.sqrt(5), np.sqrt(5), 10], # Above the line element
            [-np.sqrt(5), -np.sqrt(5), -10],
            [100, 100, 100], # Very far away
            [200, 200, -100],
        ]

        for p in points:
            assert np.allclose(self.field.current_field_at_point(p), self.axial_field.current_field_at_point(p), rtol=1e-4)

class TestCurrentLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        line = G.Path.line([-50., 0., 0.], [50., 0., 0.])
        line.name = 'line'
        
        mesh = line.mesh(mesh_size_factor=50)
         
        exc = E.Excitation(mesh, E.Symmetry.THREE_D)
        cls.current = 2.5
        exc.add_current(line=cls.current)

        cls.current_field = S.MagnetostaticSolver(exc).current_field

    def test_with_ampere_law(self):
        def correct(point):
            r = np.linalg.norm(point[1:])
            field_strength = self.current/(2*pi*r)
            angle = atan2(point[2], point[1])
            return field_strength * np.array([ 0., -np.sin(angle), np.cos(angle)])

        sample_points = [
            [0.2, 0.2, -0.2],
            [0.2, 0.2, 0.2],
            [0.2, -0.2, -0.2],
            [0.2, -0.2, 0.2],
            [-0.2, 0.2, -0.2],
            [-0.2, 0.2, 0.2],
            [-0.2, -0.2, -0.2],
            [-0.2, -0.2, 0.2],

            [0.4, 0.6, 0.],
            [0.4, 0.6, 0.2]
        ]

        for p in sample_points:
            assert np.allclose(correct(p), self.current_field.current_field_at_point(p), rtol=1e-4)

                




class TestFlatEinzelLens(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ground_top = G.Path.line([0.25, 0.0, 0.5], [0.75, 0.0, 0.5])
        lens = G.Path.line([0.25, 0.0, 0.0], [0.75, 0.0, 0.0])
        ground_bottom = G.Path.line([0.25, 0.0, -0.5], [0.75, 0.0, -0.5])
        
        ground_top.name = 'ground'
        lens.name = 'lens'
        ground_bottom.name = 'ground'
        
        # Solve radially symmetric
        mesh = (ground_top + lens + ground_bottom).mesh(mesh_size_factor=40)
        exc = E.Excitation(mesh, E.Symmetry.RADIAL)
        exc.add_voltage(ground=0, lens=1000)
        cls.field_radial = S.solve_direct(exc)
        
        # Solve three d
        ground_top = ground_top.revolve_z()
        lens = lens.revolve_z()
        ground_bottom = ground_bottom.revolve_z()
        
        mesh = (ground_top + lens + ground_bottom).mesh(mesh_size=0.1)
        exc = E.Excitation(mesh, E.Symmetry.THREE_D)
        exc.add_voltage(ground=0, lens=1000)
        cls.field = S.solve_direct(exc)
        
        cls.z = np.linspace(-0.85, 0.85, 250)
     
    def test_potential_close_to_axis(self):
        r = 0.1
        z = np.array([-0.3, 0.0, 0.3])

        pot = [self.field.potential_at_point([r, 0.0, z_]) for z_ in z]
        pot_correct = [self.field_radial.potential_at_point([r, 0.0, z_]) for z_ in z]
        assert np.allclose(pot, pot_correct, rtol=1.2e-2)
    
    def test_field_close_to_axis(self):
        r = 0.05
        z = np.array([-0.3, 0.0, 0.3])
        
        f = np.array([self.field.field_at_point([r, 0.0, z_]) for z_ in z])
        f_correct = np.array([self.field_radial.field_at_point([r, 0.0, z_]) for z_ in z])
        assert np.allclose(f, f_correct, rtol=5e-2)
    
    def test_trace_close_to_axis(self):
        r = 0.05
        z = 0.85

        tracer = self.field.get_tracer( [(-0.1, 0.1), (-0.1, 0.1), (-0.85, 0.85)] )
        tracer_radial = self.field_radial.get_tracer( [(-0.1, 0.1), (-0.1, 0.1), (-0.85, 0.85)] )
        
        p0 = np.array([r, 0.0, z])
        v0 = T.velocity_vec(100, [0, 0, -1])

        _, pos = tracer(p0, v0)
        _, pos_radial = tracer_radial(p0, v0)
         
        intersection = T.xy_plane_intersection(pos, -0.8)
        intersection_radial = T.xy_plane_intersection(pos_radial, -0.8)
         
        # We don't want to make our tests run too long, therefore the meshs size is relatively large
        # therefore we can an accuracy of only 10% here. If the mesh size is decreased the correspondence 
        # is much better.
        assert np.allclose(intersection, intersection_radial, rtol=10e-2)




