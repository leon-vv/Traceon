import unittest
from math import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, solve_ivp
from scipy.constants import mu_0

import traceon.tracing as T
import traceon.solver as S
import traceon.backend as B

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


class TestBackend(unittest.TestCase):
    
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
            EM = -0.1758820022723908 # e/m units ns and mm
            
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
        print(quad(to_integrate, 0, 2*pi)[0], current * mu_0)
        assert np.isclose(quad(to_integrate, 0, 2*pi)[0], current * mu_0)

     
    def test_electron_trajectory_in_current_loop_field(self):
        # Constants
        q = -1.602e-19  # Charge of electron in Coulombs
        m = 9.109e-31   # Mass of electron in kg
        current = 100 # Ampere on current loop
        
        # Lorentz force function
        def lorentz_force(t, y):
            v = y[3:]  # Velocity vector
            B = biot_savart_loop(current, y[:3])
            dvdt = q / m * np.cross(v, B)
            return np.hstack((v, dvdt))
        
        eV = 1e3 # Energy is 1keV
        v = sqrt(2*abs(eV*q)/m)
        
        initial_conditions = np.array([0.1, 0, 15, 0, 0, -v])
        
        t_span = (0, 1.35e-6)
        
        # Solve the differential equation
        sol = solve_ivp(lorentz_force, t_span, initial_conditions, method='DOP853', dense_output=True, rtol=1e-5, atol=1e-5)#, rtol=1e-5, atol=1e-5)
        
        # Position of electron exactly z=-10m under the loop. This is close to the optical axis
        # as the focal length of the loop is close to 10m.
        interp = CubicSpline(sol.y[2][::-1], np.array([sol.y[0], sol.y[1]]).T[::-1])
        pos_ten = interp(-10.)
        print(pos_ten)

        # Bit of a hack. We have only one currentloop but we still
        # need to define four integration points (N_QUAD_3D == 4)
        charges = S.EffectivePointCharges(
            [100],
            [[1., 0., 0., 0.]],
            [[ [1., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.]]])
        
        traceon_field = S.FieldRadialBEM(current_point_charges=charges)
         
        tracer = T.Tracer(traceon_field, ((-0.2,0.2), (-0.2, 0.2), (-20, 20)))
        pos = np.array([0., 0., 15])
        v0 = T.velocity_vec(1e3, [0., 0., -1])
         
        times, positions = tracer(pos, v0)
        print(positions)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.plot(sol.y[0], sol.y[1], sol.y[2])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
        ax.plot([0., 0.], [0., 0.], [-15, 15], linestyle='--', color='black')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.figure()
        plt.plot(sol.y[2], np.sqrt(sol.y[0]**2 + sol.y[1]**2))
        plt.show()

    


if __name__ == '__main__':
    unittest.main()



