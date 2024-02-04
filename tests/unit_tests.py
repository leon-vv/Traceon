import unittest
from math import *
import os.path as path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, solve_ivp
from scipy.constants import m_e, e, mu_0, epsilon_0

import traceon.geometry as G
import traceon.excitation as E
import traceon.tracing as T
import traceon.solver as S
import traceon.backend as B

q = -e
EM = q/m_e

# This function is known to give the correct values for the magnetic
# field for the 'unit loop' (loop with radius 1, in the xy-plane, centered around the origin).
# Very useful to test against values produced by Traceon.
# See also https://tiggerntatie.github.io/emagnet/offaxis/iloopcalculator.htm 
def biot_savart_loop(current, r_point):
    def biot_savart_integrand(t, axis):
        r_loop = np.array([np.cos(t), np.sin(t), 0])  # Position vector of the loop element
        dl = np.array([-np.sin(t), np.cos(t), 0])  # Differential element of the loop
        r = np.array(r_point) - r_loop  # Displacement vector
        db = np.cross(dl, r) / np.linalg.norm(r)**3
        return db[axis]
        
    # Magnetic field components
    Bx, _ = quad(biot_savart_integrand, 0, 2*np.pi, args=(0,))
    By, _ = quad(biot_savart_integrand, 0, 2*np.pi, args=(1,))
    Bz, _ = quad(biot_savart_integrand, 0, 2*np.pi, args=(2,))
    
    return current * mu_0 / (4 * np.pi) * np.array([Bx, By, Bz])


class TestBiotSavartLoop(unittest.TestCase):
    def test_against_online_calculator(self):
        # Test against values given by https://tiggerntatie.github.io/emagnet/offaxis/iloopcalculator.htm 
        current = 1.
         
        positions = np.array([
           [0., 0., 0.],
           [0.5, 0., 0.5],
           [0.5, 0., -0.5],
           [0.5, 0., -0.25],
           [1.001, 0., 0.],
           [1., 0., -0.001]
        ])
        
        values = np.array([
            [0., 0., 6.283185307179588e-7],
            [1.6168908407550761e-7, 0., 4.3458489359416384e-7],
            [-1.6168908407550761e-7, 0., 4.3458489359416384e-7],
            [-1.5246460125113486e-7, 0., 6.48191970028077e-7],
            [0., 0., -0.00019910189139326194],
            [-0.00019999938843237334, 0., 7.987195385643937e-7]
        ])
        
        assert np.allclose(values, [biot_savart_loop(current, p) for p in positions])
    
    def test_ampere_law(self):
        current = 2.5
        
        def to_integrate(t):
            ampere_radius = 0.5
            r_loop = np.array([1. + ampere_radius*np.cos(t), 0, ampere_radius*np.sin(t)])  # Position vector of the loop element
            dl = np.array([-ampere_radius*np.sin(t), 0, ampere_radius*np.cos(t)])  # Differential element of the loop
            field = biot_savart_loop(current, r_loop)
            # We loop counterclockwise, while positive current is defined 'into the page' which results
            # in a clockwise magnetic field. Add minus sign to compensate.
            return -np.dot(dl, field)
        
        assert np.isclose(quad(to_integrate, 0, 2*pi)[0], current * mu_0)


def potential_of_ring_arbitrary(dq, r0, z0, r, z):
    def integrand(theta):
        r_prime = np.sqrt(r**2 + r0**2 + (z0-z)**2 - 2*r*r0*np.cos(theta))
        return dq * r / r_prime # r comes from jacobian r * dtheta
    return quad(integrand, 0, 2*np.pi, epsabs=5e-13, epsrel=5e-13)[0] / (4 * np.pi * epsilon_0)


class TestPotentialRing(unittest.TestCase):
    
    def test_axial(self):
        z = np.linspace(-10, 10, 200)
        r = 1.5
        dq = 0.25
            
        # http://hyperphysics.phy-astr.gsu.edu/hbase/electric/potlin.html
        k = 1/(4*pi*epsilon_0)
        Q = dq * 2*pi*r
        correct = k * Q / np.sqrt(z**2 + r**2)
         
        pot = [potential_of_ring_arbitrary(dq, 0., z0, r, 0.) for z0 in z]
        traceon = [dq/epsilon_0*B.potential_radial_ring(0., z0, r, 0.) for z0 in z]
         
        assert np.allclose(pot, correct)
        assert np.allclose(traceon, correct)


class TestBackend(unittest.TestCase):
    
    def test_potential_radial_ring(self):
        for r0, z0, r, z in np.random.rand(10, 4):
            assert np.isclose(B.potential_radial_ring(r0, z0, r, z)/epsilon_0, potential_of_ring_arbitrary(1., r0, z0, r, z))
    
    def test_field_radial_ring(self):
        for r0, z0, r, z in np.random.rand(100, 4):
             
            dz = z0/5e4
            deriv = (potential_of_ring_arbitrary(1., r0, z0+dz, r, z) - potential_of_ring_arbitrary(1., r0, z0-dz, r, z)) / (2*dz)
            assert np.isclose(B.dz1_potential_radial_ring(r0, z0, r, z)/epsilon_0, deriv, atol=0., rtol=1e-4), (r0, z0, r, z)
            
            if r0 < 1e-3:
                continue
            
            dr = r0/5e4
            deriv = (potential_of_ring_arbitrary(1., r0+dr, z0, r, z) - potential_of_ring_arbitrary(1., r0-dr, z0, r, z)) / (2*dr)
            assert np.isclose(B.dr1_potential_radial_ring(r0, z0, r, z)/epsilon_0, deriv, atol=0., rtol=1e-4), (r0, z0, r, z)

    def test_rectangular_current_loop(self):
        with G.Geometry(G.Symmetry.RADIAL) as geom:
            points = [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
            poly = geom.add_polygon(points)
            geom.add_physical(poly, 'coil')
            geom.set_mesh_size_factor(50)
            mesh = geom.generate_triangle_mesh(False)
        
        exc = E.Excitation(mesh)
        exc.add_current(coil=5)

        field = S.solve_bem(exc)
        
        z = np.linspace(-0.5, 3.5, 300)
        r = 0.0
        
        axial_field = np.array([field.current_field_at_point(np.array([r, z_]))[1] for z_ in z])
        
        file_ = path.join(path.dirname(__file__), 'axial magnetic field.txt')
        reference_data = np.loadtxt(file_, delimiter=',')
        reference = CubicSpline(reference_data[:, 0], reference_data[:, 1]*1e-3)
         
        assert np.allclose(reference(z), axial_field, atol=0., rtol=2e-2)
        #plt.plot(z, reference(z))
        #plt.plot(z, axial_field, 'k--')
        #plt.show()
     
    def test_combine_elec_magnetic(self):
        
        for i in range(20):
            vel, elec, mag, current = np.random.rand(4, 3)
            
            result = B.combine_elec_magnetic_field(vel, elec, mag, current)
            assert np.allclose(result, elec + np.cross(vel, mag + current))
      
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

    def test_current_field_at_center(self):
        # http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/curloo.html
        R = np.linspace(0.5, 5)
        field = [B.current_field_radial_ring(0., 0., r, 0.)[1] for r in R]
        assert np.allclose(field, 1/(2*R))
        
    def test_current_axial(self):
        # http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/curloo.html
        r_ring = 55
        z_ring = 0
        
        z = np.linspace(-5, 5, 250)
        
        field_correct = r_ring**2 / (2*((z-z_ring)**2 + r_ring**2)**(3/2))
        field_z = np.array([B.current_field_radial_ring(0., z_, r_ring, z_ring)[1] for z_ in z])
        
        assert np.allclose(field_correct, field_z)
    
    def test_current_field_in_plane(self):
        # https://www.usna.edu/Users/physics/mungan/_files/documents/Scholarship/CurrentLoop.pdf
        R = 3
        I = 2.5
        B0 = I/(2*R)
        
        def field(x):
            def to_integrate(theta):
                d = x/R
                return -B0/(2*pi*d**3)*(d*cos(theta)-1)*(1+1/d**2-2/d*cos(theta))**(-3/2)
             
            return quad(to_integrate, 0, 2*pi)[0]
        
        def field_traceon(x):
            return I*B.current_field_radial_ring(x, 0., R, 0.)[1]

        x = np.linspace(0.01, 9, 10) 
        field_correct =  [field(x_)/B0 for x_ in x]
        assert np.allclose(field_correct, [field_traceon(x_)/B0 for x_ in x])
    
    def test_current_field_arbitrary(self):
                 
        positions = np.array([
            [0., 0., 0.],
            [0.5, 0., 0.],
            [0.0, 0., 0.5],
            [0.5, 0., 0.5],
            [0.5, 0., -0.5],
            [1.0, 0., 0.01],
            [1.0, 0., -0.01],
            [1.001, 0., 0.],
            [1. - 1e-3, 0., 0.],
            [1e10, 0., 0.],
        ])
        
        z_ring = 3.2
        traceon_fields = np.array([mu_0*B.current_field_radial_ring(p_[0], p_[2]+z_ring, 1., z_ring) for p_ in positions])
        correct_fields = np.array([biot_savart_loop(1., p_) for p_ in positions])
        
        assert np.allclose(traceon_fields[:, 0], correct_fields[:, 0])
        assert np.allclose(traceon_fields[:, 1], correct_fields[:, 2])
    
    def test_ampere_law(self):
        current = 2.5
        
        def to_integrate(t):
            ampere_radius = 0.5
            r_loop = np.array([1. + ampere_radius*np.cos(t), 0, ampere_radius*np.sin(t)])  # Position vector of the loop element
            dl = np.array([-ampere_radius*np.sin(t), 0, ampere_radius*np.cos(t)])  # Differential element of the loop
            field = mu_0*current*B.current_field_radial_ring(r_loop[0], r_loop[2], 1., 0.)
            # We loop counterclockwise, while positive current is defined 'into the page' which results
            # in a clockwise magnetic field. Add minus sign to compensate.
            return -np.dot(dl, [field[0], 0., field[1]])
        
        assert np.isclose(quad(to_integrate, 0, 2*pi)[0], current * mu_0)

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
        
        eff = S.EffectivePointCharges(
            [mu_0 * current],
            [[1., 0., 0., 0.]],
            [[ [1., 0., 0.],
              [1., 0., 0.],
              [1., 0., 0.],
              [1., 0., 0.]]])
        
        bounds = ((-0.4,0.4), (-0.4, 0.4), (-15, 15))
        traceon_field = S.FieldRadialBEM(current_point_charges=eff)
        tracer = T.Tracer(traceon_field, bounds, atol=1e-6)
        times, positions = tracer(initial_conditions[:3], T.velocity_vec(eV, [0, 0, -1]))
        
        interp = CubicSpline(positions[::-1, 2], np.array([positions[::-1, 0], positions[::-1, 1]]).T)
        
        assert np.allclose(interp(sol.y[2]), np.array([sol.y[0], sol.y[1]]).T)
          
    def test_current_field_ring(self):
        current = 2.5
        
        eff = S.EffectivePointCharges(
            [current],
            [[1., 0., 0., 0.]],
            [[ [1., 0., 0.],
              [1., 0., 0.],
              [1., 0., 0.],
              [1., 0., 0.]]])
         
        for p in np.random.rand(10, 3):
            field = mu_0 * B.current_field(p, eff.charges, eff.jacobians, eff.positions)
            correct = biot_savart_loop(current, p)
            assert np.allclose(field, correct)


class TestAxialInterpolation(unittest.TestCase):

    def test_current_axial(self):
        # http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/curloo.html
        r_ring = 2
        z_ring = 0
        
        z = np.linspace(-5, 5, 250)
        dz = z - z_ring
         
        pot_correct = -dz / (2*np.sqrt(dz**2 + r_ring**2))
        field_correct = r_ring**2 / (2*((z-z_ring)**2 + r_ring**2)**(3/2))
        
        pot_z = np.array([B.current_potential_axial_radial_ring(z_, r_ring, z_ring) for z_ in z])
        field_z = np.array([B.current_field_radial_ring(0., z_, r_ring, z_ring)[1] for z_ in z])
        
        assert np.allclose(pot_correct, pot_z)
        assert np.allclose(field_correct, field_z)
        
        numerical_derivative = CubicSpline(z, pot_correct).derivative()(z)
        assert np.allclose(-numerical_derivative, field_z)
    
    def test_current_loop(self):
        current = 2.5
        
        eff = S.EffectivePointCharges(
            [current],
            [[1., 0., 0., 0.]],
            [[ [1., 0., 0.],
              [1., 0., 0.],
              [1., 0., 0.],
              [1., 0., 0.]]])
        
        traceon_field = S.FieldRadialBEM(current_point_charges=eff)
         
        z = np.linspace(-6, 6, 250)
        pot = [traceon_field.current_potential_axial(z_) for z_ in z]
        field = [traceon_field.current_field_at_point(np.array([0.0, z_]))[1] for z_ in z]

        numerical_derivative = CubicSpline(z, pot).derivative()(z)
        
        assert np.allclose(field, -numerical_derivative)
    
    def test_derivatives(self):
        current = 2.5
        
        eff = S.EffectivePointCharges(
            [current],
            [[1., 0., 0., 0.]],
            [[ [1., 0., 0.],
              [1., 0., 0.],
              [1., 0., 0.],
              [1., 0., 0.]]])
        
        traceon_field = S.FieldRadialBEM(current_point_charges=eff)
         
        z = np.linspace(-6, 6, 500)
        derivatives = traceon_field.get_current_axial_potential_derivatives(z)

        pot = [traceon_field.current_potential_axial(z_) for z_ in z]
        
        assert np.allclose(pot, derivatives[:, 0])

        interp = CubicSpline(z, pot)
        d1 = interp.derivative()
        assert np.allclose(d1(z), derivatives[:, 1])
        
        for i in range(0, derivatives.shape[1]-1):
            interp = CubicSpline(z, derivatives[:, i])
            deriv = interp.derivative()(z)
            #plt.plot(z, deriv)
            #plt.plot(z, derivatives[:, i+1], linestyle='dashed')
            #plt.show()
            assert np.allclose(deriv, derivatives[:, i+1], atol=1e-5, rtol=5e-3)

    def test_interpolation_current_loop(self):
        current = 2.5
        
        eff = S.EffectivePointCharges(
            [current],
            [[1., 0., 0., 0.]],
            [[ [1., 0., 0.],
              [1., 0., 0.],
              [1., 0., 0.],
              [1., 0., 0.]]])
        
        traceon_field = S.FieldRadialBEM(current_point_charges=eff)
        interp = traceon_field.axial_derivative_interpolation(-5, 5, N=300)

        z = interp.z[1:-1]

        pot = [traceon_field.current_potential_axial(z_) for z_ in z]
        pot_interp = [interp.magnetostatic_potential_at_point(np.array([0., z_])) for z_ in z]

        assert np.allclose(pot, pot_interp)
         
        r = np.linspace(-0.5, 0.5, 5)
        
        for r_ in r:
            field = [traceon_field.magnetostatic_field_at_point(np.array([r_, z_]))[1] for z_ in z]
            field_interp = [interp.magnetostatic_field_at_point(np.array([r_, z_]))[1] for z_ in z]
            assert np.allclose(field, field_interp, atol=1e-3, rtol=5e-3)

if __name__ == '__main__':
    unittest.main()
















