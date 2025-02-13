import unittest
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import m_e, e, mu_0, epsilon_0
from scipy.interpolate import CubicSpline

import voltrace.backend as B
import voltrace as v

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
            return acceleration() / EM, np.zeros(3)

        bounds = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, np.sqrt(12)+1))
        times, positions = B.trace_particle(np.zeros( (3,) ), np.array([0., 0., 3.]), EM, B.wrap_field_fun(field), bounds, 1e-10)

        correct_x = 3/2*times**2
        correct_z = 3*times

        assert np.allclose(correct_x, positions[:, 0])
        assert np.allclose(correct_z, positions[:, 2])
    
    def test_tracing_multiple_constant_acceleration(self):
        # Same test as above but using a custom field and the trace_many function

        class ConstantField(v.Field):
            def is_electrostatic(self):
                return True
            def is_magnetostatic(self):
                return False
            def magnetostatic_field_at_local_point(self, point):
                return np.zeros(3)
            def magnetostatic_potential_at_local_point(self, point):
                return 0.0
            def electrostatic_potential_at_local_point(self):
                return 0.0
            def electrostatic_field_at_local_point(self, point):
                acceleration = np.array([3., 0., 0.])
                return acceleration * m_e/(-e)
        
        bounds = ((-2.0, 2.0), (-2.0, 2.0), (-2.0, np.sqrt(12)+1))

        tracer = v.Tracer(ConstantField(), bounds)
         
        result = tracer.trace_multiple([np.zeros(3), np.zeros(3)], [0., 0., 3.])
        
        for times, positions in result:
            correct_x = 3/2*times**2
            correct_z = 3*times
            
            assert np.allclose(correct_x, positions[:, 0])
            assert np.allclose(correct_z, positions[:, 2])
    
    def test_tracing_helix_against_scipy(self):
        def acceleration(_, y):
            velocity = y[3:]
            B = np.array([0, 0, 1])
            return np.hstack( (velocity, np.cross(velocity, B)) )
         
        def voltrace_acc(pos, vel):
            return np.zeros(3), np.array([0, 0, 1])/(mu_0*EM)
         
        p0 = np.zeros(3)
        v0 = np.array([0., 1, -1.])
        
        bounds = ((-5.0, 5.0), (-5.0, 5.0), (-40.0, 10.0))
        times, positions = B.trace_particle(p0, v0, EM, B.wrap_field_fun(voltrace_acc), bounds, 1e-10)
        
        sol = solve_ivp(acceleration, (0, 30), np.hstack( (p0, v0) ), method='DOP853', rtol=1e-10, atol=1e-10)

        interp = CubicSpline(positions[::-1, 2], np.array([positions[::-1, 0], positions[::-1, 1]]).T)

        assert np.allclose(interp(sol.y[2]), np.array([sol.y[0], sol.y[1]]).T)
    
    def test_tracing_helix_against_scipy_custom_field(self):
        def acceleration(_, y):
            velocity = y[3:]
            B = np.array([0, 0, 1])
            return np.hstack( (velocity, np.cross(velocity, B)) )

        class CustomField(v.Field):
            def magnetostatic_field_at_local_point(self, point):
                return np.array([0, 0, 1])/(mu_0*EM)
            
            def magnetostatic_potential_at_local_point(self, point):
                return 1./(mu_0*EM)
            
            def electrostatic_field_at_local_point(self, point):
                return np.array([0, 0, 0])

            def electrostatic_potential_at_local_point(self):
                return 0.0

            def is_magnetostatic(self):
                return True

            def is_electrostatic(self):
                return False

        p0 = np.zeros(3)
        v0 = np.array([0., 1, -1.])
        
        bounds = ((-5.0, 5.0), (-5.0, 5.0), (-40.0, 10.0))
        tracer = v.Tracer(CustomField(), bounds)
        
        # Note that we transform velocity to eV, since it's being converted back to m/s in the Tracer.__call__ function
        times, positions = tracer(p0, v0, atol=1e-10)
         
        sol = solve_ivp(acceleration, (0, 30), np.hstack( (p0, v0) ), method='DOP853', rtol=1e-10, atol=1e-10)

        interp = CubicSpline(positions[::-1, 2], np.array([positions[::-1, 0], positions[::-1, 1]]).T)
        
        assert np.allclose(interp(sol.y[2]), np.array([sol.y[0], sol.y[1]]).T)

    
    def test_tracing_against_scipy_current_loop(self):
        # Constants
        current = 100 # Ampere on current loop
        
        # Lorentz force function
        def lorentz_force(_, y):
            velocity = y[3:]  # Velocity vector
            B = biot_savart_loop(current, y[:3])
            dvdt = EM * np.cross(velocity, B)
            return np.hstack((velocity, dvdt))
        
        eV = 1e3 # Energy is 1keV
        speed = sqrt(2*abs(eV*q)/m_e)
         
        initial_conditions = np.array([0.1, 0, 15, 0, 0, -speed])
        sol = solve_ivp(lorentz_force, (0, 1.35e-6), initial_conditions, method='DOP853', rtol=1e-6, atol=1e-6)

        eff = get_ring_effective_point_charges(current, 1.)
          
        bounds = ((-0.4,0.4), (-0.4, 0.4), (-15, 15))
        voltrace_field = v.FieldRadialBEM(current_point_charges=eff)
        tracer = voltrace_field.get_tracer(bounds)
        times, positions = tracer(initial_conditions[:3], v.velocity_vec(eV, [0, 0, -1]), atol=1e-6)
        
        interp = CubicSpline(positions[::-1, 2], np.array([positions[::-1, 0], positions[::-1, 1]]).T)
        
        assert np.allclose(interp(sol.y[2]), np.array([sol.y[0], sol.y[1]]).T)
    
    def test_interpolated_tracing_against_scipy_current_loop(self):
        # Constants
        current = 100 # Ampere on current loop
        
        # Lorentz force function
        def lorentz_force(_, y):
            velocity = y[3:]  # Velocity vector
            B = biot_savart_loop(current, y[:3])
            dvdt = EM * np.cross(velocity, B)
            return np.hstack((velocity, dvdt))
        
        eV = 1e3 # Energy is 1keV
        speed = sqrt(2*abs(eV*q)/m_e)
         
        initial_conditions = np.array([0.05, 0, 15, 0, 0, -speed])
        sol = solve_ivp(lorentz_force, (0, 1.35e-6), initial_conditions, method='DOP853', rtol=1e-6, atol=1e-6)
         
        eff = get_ring_effective_point_charges(current, 1.)
         
        bounds = ((-0.4,0.4), (-0.4, 0.4), (-15, 15))

        field = v.FieldRadialBEM(current_point_charges=eff)
        axial_field = v.FieldRadialAxial(field, -15, 15, N=500)
         
        tracer = axial_field.get_tracer(bounds)
        times, positions = tracer(initial_conditions[:3], v.velocity_vec(eV, [0, 0, -1]), atol=1e-6)
         
        interp = CubicSpline(positions[::-1, 2], np.array([positions[::-1, 0], positions[::-1, 1]]).T)
        
        assert np.allclose(interp(sol.y[2]), np.array([sol.y[0], sol.y[1]]).T, atol=1e-4, rtol=5e-5)

       
    def test_plane_intersection(self):
        p = np.array([
            [3, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0.]]);

        result = v.plane_intersection(p, np.zeros(3), np.array([4.0,0.0,0.0]))
        assert np.allclose(result, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        p = np.array([
            [2, 2, 2, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [-1, -1, -1, 0, 0, 0.]]);

        result = v.plane_intersection(p, np.zeros(3), np.array([2.0,2.0,2.0]))
        assert np.allclose(result, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        result = v.plane_intersection(p, np.zeros(3), np.array([-1.0,-1.0,-1.0]))
        assert np.allclose(result, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        p = np.array([
            [2, 2, 2, 0, 0, 0],
            [2, 2, 1, 0, 1, 0],
            [2, 2, -3, 0, 0, 0.]]);

        result = v.plane_intersection(p, np.array([2.,2.,1.0]), np.array([0.0,0.0,1.0]))
        assert np.allclose(result, np.array([2.0, 2.0, 1.0, 0.0, 1.0, 0.0]))

        result = v.plane_intersection(p, np.array([2.,2.,1.0]), np.array([0.0,0.0,-1.0]))
        assert result is not None and np.allclose(result, np.array([2.0, 2.0, 1.0, 0.0, 1.0, 0.0]))

        p = np.array([
            [0., 0, -3, 0, 0, 0],
            [0., 0, 9, 0, 0, 0]])

        result = v.plane_intersection(p, np.array([0.,1.,0.0]), np.array([1.0,1.0,1.0]))
        assert np.allclose(result, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        result = v.plane_intersection(p, np.array([0.,1.,0.0]), -np.array([1.0,1.0,1.0]))
        assert np.allclose(result, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))

        p = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        r = v.xy_plane_intersection(p, 0.5)[0]
        assert np.isclose(r, 1.0)

        p = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0]])
        r = v.xy_plane_intersection(p, -0.5)[0]
        assert np.isclose(r, 1.0)

        p = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, -2.0, 0.0, 0.0, 0.0]])
        r = v.xy_plane_intersection(p, -1.5)[0]
        assert np.isclose(r, 1.5)

        p = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 2.0, 0.0, 0.0, 0.0]])
        r = v.xy_plane_intersection(p, 1.75)[0]
        assert np.isclose(r, 3.25)

        p = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 2.0, 0.0, 0.0, 0.0]])
        y = v.xy_plane_intersection(p, 1.75)[1]
        assert np.isclose(y, 3.25)
    
    def test_cyclotron_radius(self):
         
        x0 = np.array([1., 0., 0.])
        v0 = np.array([0., 1., 0.])
        bounds = ((-2., 2.), (0., 2.), (-2., 2.))
         
        def field(pos, vel):
            return np.zeros(3), np.array([0, 0, -1.])/(EM*mu_0)
        
        times, positions = B.trace_particle(x0, v0, EM, B.wrap_field_fun(field), bounds, 1e-10)

        # Map y-position to state
        interp = CubicSpline(positions[-10:, 1][::-1], positions[-10:][::-1])
        
        # State of particle at intersection of y-axis
        y_intersection = interp(0.)
        assert np.allclose(y_intersection, np.array([-1., 0, 0, 0, -1, 0]))

    def test_superposition_tracing(self):
        pos = v.Path.rectangle_xz(0.1,1,1, 1.5)
        neg = v.Path.rectangle_xz(0.1,1,-1.5, -1)
        neg.name='neg'
        pos.name='pos'

        mesh = (neg + pos).mesh(mesh_size=1)

        excitation = v.Excitation(mesh, v.Symmetry.RADIAL)
        excitation.add_voltage(pos=1, neg=-1)
        field = v.solve_direct(excitation)

        # superposition of field and field should be the same as doubling the field strength
        field_superposition = v.FieldSuperposition([field, field])
        field_double = 2 * field

        tracer_sup = field_superposition.get_tracer([[-2,2], [-2,2], [-2,2]])
        tracer_double = field_double.get_tracer([[-2,2], [-2,2], [-2,2]])
        start = np.array([0.001, 0, 2])
        velocity = v.velocity_vec(10, [0, 0, -1])

        _, trajectory_sup = tracer_sup(start, velocity)
        _, trajectory_double = tracer_double(start, velocity)

        assert np.allclose(trajectory_sup, trajectory_double)

        #symmetry in xy plane so field should be zero everywhere
        # so trajectory should be straight line
        field_sup2 = v.FieldSuperposition([field, field.rotate(Ry=np.pi)])

        tracer_sup2= field_sup2.get_tracer([[-2,2], [-2,2], [-2,2]])
        _, trajectory = tracer_sup2(start, velocity)
        assert np.allclose(v.plane_intersection(np.array(trajectory), np.array([0.,0.,0.]), np.array([0.,0.,-1.]))[:3], np.array([0.001, 0, 0]))
        assert np.allclose(v.plane_intersection(np.array(trajectory), np.array([0.,0.,-1.]), np.array([0.,0.,-1.]))[:3], np.array([0.001, 0, -1]))
        assert np.allclose(v.plane_intersection(np.array(trajectory), np.array([0.,0.,-1.5]), np.array([0.,0.,-1.]))[:3], np.array([0.001, 0, -1.5]))

