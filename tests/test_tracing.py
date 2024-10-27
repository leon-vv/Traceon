import unittest
from math import sqrt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import m_e, e, mu_0, epsilon_0
from scipy.interpolate import CubicSpline

import traceon.backend as B
import traceon.solver as S
import traceon.tracing as T
from traceon.interpolation import FieldRadialAxial

from tests.test_radial_ring import biot_savart_loop
from tests.test_radial import get_ring_effective_point_charges

q = -e
EM = q/m_e


class TestTracing(unittest.TestCase):
    def test_tracing_constant_acceleration(self):
        def acceleration(*_):
            return np.array([3., 0., 0.])

        def field(*_):
            acceleration_x = 3 # mm/ns
            return acceleration() / EM

        bounds = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, np.sqrt(12)+1))
        times, positions = B.trace_particle(np.zeros( (3,) ), np.array([0., 0., 3.]), field, bounds, 1e-10)

        correct_x = 3/2*times**2
        correct_z = 3*times

        assert np.allclose(correct_x, positions[:, 0])
        assert np.allclose(correct_z, positions[:, 2])

    def test_tracing_helix_against_scipy(self):
        def acceleration(_, y):
            v = y[3:]
            B = np.array([0, 0, 1])
            return np.hstack( (v, np.cross(v, B)) )
         
        def traceon_acc(*y):
            return acceleration(0., y)[3:] / EM
        
        p0 = np.zeros(3)
        v0 = np.array([0., 1, -1.])
        
        bounds = ((-5.0, 5.0), (-5.0, 5.0), (-40.0, 10.0))
        times, positions = B.trace_particle(p0, v0, traceon_acc, bounds, 1e-10)
        
        sol = solve_ivp(acceleration, (0, 30), np.hstack( (p0, v0) ), method='DOP853', rtol=1e-10, atol=1e-10)

        interp = CubicSpline(positions[::-1, 2], np.array([positions[::-1, 0], positions[::-1, 1]]).T)

        assert np.allclose(interp(sol.y[2]), np.array([sol.y[0], sol.y[1]]).T)
    
    def test_tracing_against_scipy_current_loop(self):
        # Constants
        current = 100 # Ampere on current loop
        
        # Lorentz force function
        def lorentz_force(_, y):
            v = y[3:]  # Velocity vector
            B = biot_savart_loop(current, y[:3])
            dvdt = EM * np.cross(v, B)
            return np.hstack((v, dvdt))
        
        eV = 1e3 # Energy is 1keV
        v = sqrt(2*abs(eV*q)/m_e)
         
        initial_conditions = np.array([0.1, 0, 15, 0, 0, -v])
        sol = solve_ivp(lorentz_force, (0, 1.35e-6), initial_conditions, method='DOP853', rtol=1e-6, atol=1e-6)

        eff = get_ring_effective_point_charges(current, 1.)
          
        bounds = ((-0.4,0.4), (-0.4, 0.4), (-15, 15))
        traceon_field = S.FieldRadialBEM(current_point_charges=eff)
        tracer = traceon_field.get_tracer(bounds)
        times, positions = tracer(initial_conditions[:3], T.velocity_vec(eV, [0, 0, -1]), atol=1e-6)
        
        interp = CubicSpline(positions[::-1, 2], np.array([positions[::-1, 0], positions[::-1, 1]]).T)
        
        assert np.allclose(interp(sol.y[2]), np.array([sol.y[0], sol.y[1]]).T)
    
    def test_interpolated_tracing_against_scipy_current_loop(self):
        # Constants
        current = 100 # Ampere on current loop
        
        # Lorentz force function
        def lorentz_force(_, y):
            v = y[3:]  # Velocity vector
            B = biot_savart_loop(current, y[:3])
            dvdt = EM * np.cross(v, B)
            return np.hstack((v, dvdt))
        
        eV = 1e3 # Energy is 1keV
        v = sqrt(2*abs(eV*q)/m_e)
         
        initial_conditions = np.array([0.05, 0, 15, 0, 0, -v])
        sol = solve_ivp(lorentz_force, (0, 1.35e-6), initial_conditions, method='DOP853', rtol=1e-6, atol=1e-6)
         
        eff = get_ring_effective_point_charges(current, 1.)
         
        bounds = ((-0.4,0.4), (-0.4, 0.4), (-15, 15))

        field = S.FieldRadialBEM(current_point_charges=eff)
        axial_field = FieldRadialAxial(field, -15, 15, N=500)
         
        tracer = axial_field.get_tracer(bounds)
        times, positions = tracer(initial_conditions[:3], T.velocity_vec(eV, [0, 0, -1]), atol=1e-6)
         
        interp = CubicSpline(positions[::-1, 2], np.array([positions[::-1, 0], positions[::-1, 1]]).T)
        
        assert np.allclose(interp(sol.y[2]), np.array([sol.y[0], sol.y[1]]).T, atol=1e-4, rtol=5e-5)
    
    def test_plane_intersection(self):
        p = np.array([
            [3, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0.]]);

        result = T.plane_intersection(p, np.zeros(3), np.array([4.0,0.0,0.0]))
        assert np.allclose(result, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        p = np.array([
            [2, 2, 2, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [-1, -1, -1, 0, 0, 0.]]);

        result = T.plane_intersection(p, np.zeros(3), np.array([2.0,2.0,2.0]))
        assert np.allclose(result, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        result = T.plane_intersection(p, np.zeros(3), np.array([-1.0,-1.0,-1.0]))
        assert np.allclose(result, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        p = np.array([
            [2, 2, 2, 0, 0, 0],
            [2, 2, 1, 0, 1, 0],
            [2, 2, -3, 0, 0, 0.]]);

        result = T.plane_intersection(p, np.array([2.,2.,1.0]), np.array([0.0,0.0,1.0]))
        assert np.allclose(result, np.array([2.0, 2.0, 1.0, 0.0, 1.0, 0.0]))

        result = T.plane_intersection(p, np.array([2.,2.,1.0]), np.array([0.0,0.0,-1.0]))
        assert result is not None and np.allclose(result, np.array([2.0, 2.0, 1.0, 0.0, 1.0, 0.0]))

        p = np.array([
            [0., 0, -3, 0, 0, 0],
            [0., 0, 9, 0, 0, 0]])

        result = T.plane_intersection(p, np.array([0.,1.,0.0]), np.array([1.0,1.0,1.0]))
        assert np.allclose(result, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        result = T.plane_intersection(p, np.array([0.,1.,0.0]), -np.array([1.0,1.0,1.0]))
        assert np.allclose(result, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))

        p = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0]])
        r = T.xy_plane_intersection(p, 0.5)[0]
        assert np.isclose(r, 1.0)

        p = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0]])
        r = T.xy_plane_intersection(p, -0.5)[0]
        assert np.isclose(r, 1.0)

        p = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [2.0, -2.0, 0.0, 0.0]])
        r = T.xy_plane_intersection(p, -1.5)[0]
        assert np.isclose(r, 1.5)

        p = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [4.0, 2.0, 0.0, 0.0]])
        r = T.xy_plane_intersection(p, 1.75)[0]
        assert np.isclose(r, 3.25)

        p = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 2.0, 0.0, 0.0, 0.0]])
        y = T.xy_plane_intersection(p, 1.75)[1]
        assert np.isclose(y, 3.25)
    
    def test_cyclotron_radius(self):
         
        x0 = np.array([1., 0., 0.])
        v0 = np.array([0., 1., 0.])
        bounds = ((-2., 2.), (0., 2.), (-2., 2.))
         
        def field(x, y, z, vx, vy, vz):
            p, v = np.array([x,y,z]), np.array([vx, vy, vz])
            mag_field = np.array([0, 0, -1.])
            return np.cross(v, mag_field) / EM # Return acceleration
        
        times, positions = B.trace_particle(x0, v0, field, bounds, 1e-10)

        # Map y-position to state
        interp = CubicSpline(positions[-10:, 1][::-1], positions[-10:][::-1])
        
        # State of particle at intersection of y-axis
        y_intersection = interp(0.)
        assert np.allclose(y_intersection, np.array([-1., 0, 0, 0, -1, 0]))
     
