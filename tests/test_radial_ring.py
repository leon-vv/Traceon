import unittest
from math import pi, cos

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.constants import e, m_e, mu_0, epsilon_0
from scipy.integrate import quad

import traceon.backend as B
import traceon.logging as logging

logging.set_log_level(logging.LogLevel.SILENT)



def potential_radial_exact_integrated(v0, v1, target):
    assert v0.shape == (2,) and v1.shape == (2,) and target.shape == (2,)
    
    def integrand(alpha, phi):
        r, z = v0 + (v1-v0)*alpha
        r_vec = np.array([r*cos(phi), r*sin(phi), z])
        distance = np.linalg.norm(r_vec - np.array([target[0], 0.0, target[1]]))

        jacobian = r
        
        return 1/(pi*distance) * jacobian

    length = np.linalg.norm(v0-v1)
    
    return dblquad(integrand, 0, 1, 0, 2*pi, epsabs=1e-10, epsrel=1e-10)[0] * length


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
    Bx, _ = quad(biot_savart_integrand, 0, 2*np.pi, args=(0,), epsabs=5e-13)
    By, _ = quad(biot_savart_integrand, 0, 2*np.pi, args=(1,), epsabs=5e-13)
    Bz, _ = quad(biot_savart_integrand, 0, 2*np.pi, args=(2,), epsabs=5e-13)
    
    return current * mu_0 / (4 * np.pi) * np.array([Bx, By, Bz])

def magnetic_field_of_loop(current, radius, point):
    return biot_savart_loop(current, point/radius)/radius


def potential_of_ring_arbitrary(dq, r0, z0, r, z):
    def integrand(theta):
        r_prime = np.sqrt(r**2 + r0**2 + (z0-z)**2 - 2*r*r0*np.cos(theta))
        return dq * r / r_prime # r comes from jacobian r * dtheta
    return quad(integrand, 0, 2*np.pi, epsabs=5e-13, epsrel=5e-13)[0] / (4 * np.pi * epsilon_0)


class TestRadialRing(unittest.TestCase):
    
    def test_potential_radial_ring(self):
        for r0, z0, r, z in np.random.rand(10, 4):
            assert np.isclose(B.potential_radial_ring(r0, z0, r, z)/epsilon_0, potential_of_ring_arbitrary(1., r0, z0, r, z))
    
    def test_potential_field_radial_single_ring(self):
        rs = 1 # Source point
        zs = 0

        r = np.linspace(1.1, 2, 10000)
        pot = [B.potential_radial_ring(r_, zs, rs, zs) for r_ in r]
        deriv = [B.dr1_potential_radial_ring(r_, zs, rs, zs) for r_ in r]
        assert np.allclose(CubicSpline(r, pot)(r, 1), deriv, atol=0., rtol=1e-9)
        
        z = np.linspace(0.1, 1, 10000)
        pot = [B.potential_radial_ring(rs, z_, rs, zs) for z_ in z]
        deriv = [B.dz1_potential_radial_ring(rs, z_, rs, zs) for z_ in z]
        assert np.allclose(CubicSpline(z, pot)(z, 1), deriv, atol=0., rtol=1e-9)
    
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
    
    def test_magnetic_field_of_loop_against_backend2(self):
        N = 50
        test_cases = 10*np.random.rand(N, 4)
        
        for x0, y0, x, y in test_cases:
            traceon_field = mu_0*B.current_field_radial_ring(x0, y0, x, y)
            correct_field = magnetic_field_of_loop(1., x, np.array([x0, 0., y0-y]))
            
            assert np.isclose(traceon_field[0], correct_field[0], atol=1e-10)
            assert np.isclose(traceon_field[1], correct_field[2], atol=1e-10)   
    
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
        # http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/curloo.html
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
        
        assert np.allclose(traceon_fields[:, 0], correct_fields[:, 0], atol=1e-10)
        assert np.allclose(traceon_fields[:, 1], correct_fields[:, 2], atol=1e-10)
    
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
    
    def test_magnetic_field_of_loop_against_online_calculator(self):
        # Current, radius, x, y, z
        test_cases = np.array([
            [0.5, 2.5, 2.5, 0, 1.5],
            [0.5, 2.5, 1.0, 0, 1.0],
            [1.0, 2.5, 2.5, 0., 4.],
            [1.25, 5, 1.0, 1.0, 1.0]])
        
        theta = np.pi/4
        rad_vector = np.array([np.cos(theta), np.sin(theta), 0.])
        axial_vector = np.array([0., 0., 1.])
         
        correct = np.array([
            [5.1469970601554224e-8, 0., 3.02367391882467e-8],
            [2.4768749699767386e-8, 0., 1.0287875364876885e-7],
            [1.7146963475789695e-8, 0., 2.083787779958955e-8],
            rad_vector * 1.3835602159875731e-8 + axial_vector * 1.553134392842064e-7])
        
        for t, c in zip(test_cases, correct):
            assert np.allclose(magnetic_field_of_loop(t[0], t[1], t[2:]), c, atol=1e-9), (magnetic_field_of_loop(t[0], t[1], t[2:]), c)

    
            
    
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




