import unittest
from math import pi, sqrt
import os.path as path

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.constants import epsilon_0, mu_0
from scipy.interpolate import CubicSpline

import traceon.geometry as G
import traceon.solver as S
import traceon.excitation as E
import traceon.tracing as T
import traceon.backend as B
import traceon.logging as logging

logging.set_log_level(logging.LogLevel.SILENT)

class TestFlatEinzelLens(unittest.TestCase):

    def setUp(self):
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
        self.field_radial = S.solve_direct(exc)
        
        # Solve three d
        ground_top = ground_top.revolve_z()
        lens = lens.revolve_z()
        ground_bottom = ground_bottom.revolve_z()
        
        mesh = (ground_top + lens + ground_bottom).mesh(mesh_size=0.1)
        exc = E.Excitation(mesh, E.Symmetry.THREE_D)
        exc.add_voltage(ground=0, lens=1000)
        self.field = S.solve_direct(exc)
        
        self.z = np.linspace(-0.85, 0.85, 250)
     
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




