import unittest, time
from math import *
import os.path as path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, solve_ivp, dblquad
from scipy.constants import m_e, e, mu_0, epsilon_0
from scipy.special import ellipe, ellipk, ellipkm1

from traceon import focus as F
import traceon.geometry as G
import traceon.excitation as E
import traceon.tracing as T
import traceon.solver as S
import traceon.backend as B
import traceon.plotting as P
import traceon.logging as logging
import traceon.fast_multipole_method as FMM

logging.set_log_level(logging.LogLevel.SILENT)

def rand(*shape, min=-100, max=100):
    return min + np.random.rand(*shape)*(max-min)

def potential_exact(a, b, target, points):
    v0, v1, v2 = points
    p = v0 + a*(v1-v0) + b*(v2-v0)
     
    return 1/np.linalg.norm(p - target)

def flux_exact(a, b, target, points, normal):
    v0, v1, v2 = points
    x, y, z = v0 + a*(v1-v0) + b*(v2-v0)
    x0, y0, z0 = target
    denominator = ((x-x0)**2 + (y-y0)**2 + (z-z0)**2)**(3/2)
    dx = -(x-x0)/denominator
    dy = -(y-y0)/denominator
    dz = -(z-z0)/denominator
    return np.dot(normal, [dx, dy, dz])

def potential_exact_integrated(v0, v1, v2, target):
    area = np.linalg.norm(np.cross(v1-v0, v2-v0))/2
    return dblquad(potential_exact, 0, 1, 0, lambda x: 1-x, epsabs=1e-12, epsrel=1e-12, args=(target, (v0, v1, v2)))[0] * (2*area)

def flux_exact_integrated(v0, v1, v2, target, normal):
    area = np.linalg.norm(np.cross(v1-v0, v2-v0))/2
    return dblquad(flux_exact, 0, 1, 0, lambda x: 1-x, epsabs=5e-14, epsrel=5e-14, args=(target, (v0, v1, v2), normal))[0] * (2*area)

def get_ring_effective_point_charges(current, r):
    return S.EffectivePointCharges(
        [current],
        [ [1.] + ([0.]*(B.N_TRIANGLE_QUAD-1)) ],
        [ [[r, 0., 0.]] * B.N_TRIANGLE_QUAD ])

class TestFMM(unittest.TestCase):
    
    def test_matrix_application_edwards2007(self):
        with G.Geometry(G.Symmetry.THREE_D) as geom:
            points = [ [0, 0], [0, 5], [12, 5], [12, 15],
                [0, 15], [0, 20], [20, 20], [20, 0] ]
            
            geom.set_mesh_size_factor(15)
            
            points = [geom.add_point([p[0], 0.0, p[1]]) for p in points]
            
            l1 = geom.add_line(points[1], points[2])
            l2 = geom.add_line(points[2], points[3])
            l3 = geom.add_line(points[3], points[4])
            
            l4 = geom.add_line(points[0], points[-1])
            l5 = geom.add_line(points[-3], points[-2])
            l6 = geom.add_line(points[-2], points[-1])
            
            inner = G.revolve_around_optical_axis(geom, [l1, l2, l3])
            boundary = G.revolve_around_optical_axis(geom, [l4, l5, l6])
            
            geom.add_physical(inner, 'inner')
            geom.add_physical(boundary, 'boundary')
                    
            mesh = geom.generate_triangle_mesh()
        
        exc = E.Excitation(mesh)
        exc.add_voltage(inner=10, boundary=0)
        
        solver = S.ElectrostaticSolver(exc)
        
        dielectric_indices = solver.get_flux_indices()
        dielectric_values = solver.excitation_values[dielectric_indices]
        dielectric_factors = np.array([backend.flux_density_to_charge_factor(k) for k in dielectric_values])
         
        charges = rand(len(mesh.triangles), min=-3, max=3)
        
        geom_fortran = FMM.get_geometry_in_fortran_layout(solver.vertices)
        result_fmm = FMM.apply_matrix(charges, geom_fortran, 5, dielectric_indices, dielectric_factors)
         
        result_direct =  solver.get_matrix() @ charges
         
        assert np.allclose(result_fmm, result_direct, atol=0.0, rtol=1e-6)



class TestElliptic(unittest.TestCase):
    
    def test_ellipk(self):
        x = np.linspace(-1, 1, 40)[1:-1]
        # ellipk is slightly inaccurate, but is not used in electrostatic
        # solver. Only ellipkm1 is used
        assert np.allclose(B.ellipk(x), ellipk(x), atol=0., rtol=1e-5)
    
    def test_ellipe(self):
        x = np.linspace(-1, 1, 40)[1:-1]
        assert np.allclose(B.ellipe(x), ellipe(x), atol=0., rtol=1e-12)
 
    def test_ellipkm1_big(self):
        x = np.linspace(0, 1)[1:-1]
        assert np.allclose(ellipkm1(x), B.ellipkm1(x), atol=0., rtol=1e-12)
 
    def test_ellipkm1_small_many(self):
        x = np.linspace(1, 100, 5)
        assert np.allclose(ellipkm1(10**(-x)), B.ellipkm1(10**(-x)), atol=0., rtol=1e-12)
    
    def test_ellipem1_small_many(self):
        x = np.linspace(1, 100, 5)
        assert np.allclose(ellipe(1 - 10**(-x)), B.ellipem1(10**(-x)), atol=0., rtol=1e-12)


class TestTriangleContribution(unittest.TestCase):

    def test_self_potential_quadrants(self):
        def test(a, b, c):
            v0, v1, v2 = np.array([
                    [0., 0., 0.],
                    [a, 0., 0.],
                    [b, c, 0.]])
            
            target = np.array([0., 0., 0.])
            correct = potential_exact_integrated(v0, v1, v2, target)
            approx = B.self_potential_triangle_v0(v0, v1, v2)
            assert np.isclose(approx, correct, atol=0., rtol=1e-15)
        
        test(1, 0, 1) # right triangle quadrant 1 
        test(1, 1, 1) # right triangle quadrant 1
        test(-1, 0, 1) # right triangle quadrant 2
        test(-1, -1, 1) # right triangle quadrant 2
        test(-1, 0, -1) # right triangle quadrant 3
        test(-1, -1, -1) # right triangle quadrant 3
        test(1, 0, -1) # right triangle quadrant 4
        test(1, 1, -1) # right triangle quadrant 4
    
    def test_self_potential_random(self):
        def test(a, b, c):
            v0, v1, v2 = np.array([
                    [0., 0., 0.],
                    [a, 0., 0.],
                    [b, c, 0.]])
            
            target = (v0+v1+v2)/3
             
            correct = potential_exact_integrated(v0, v1, v2, target)
            approx = B.self_potential_triangle(v0, v1, v2, target)
            assert np.isclose(approx, correct, atol=0., rtol=1e-12), (a,b,c)

        for (a,b,c) in rand(3, 3):
            test(a,b,c)

    def test_derivative_x_quadrants(self):
        def test(a, b, c, z0):
            x0 = 0.
            v0, v1, v2 = np.array([
                    [x0, 0., 0.],
                    [x0 + a, 0., 0.],
                    [x0 + b, c, 0.]])
            
            target = np.array([0., 0., z0])
            normal = np.array([1., 0., 0.])
            
            correct = flux_exact_integrated(v0, v1, v2, target, normal)
            approx = B.flux_triangle(v0, v1, v2, target, normal)
            assert np.isclose(correct, approx, atol=0., rtol=1e-12), (x0, a,b,c, z0)
        
        test(1, 0, 1, 0.1) # right triangle quadrant 1 
        test(1, 1, 1, 0.1) # right triangle quadrant 1
        test(1, 0, -1, 0.1) # right triangle quadrant 4
        test(1, 1, -1, 0.1) # right triangle quadrant 4
    
    def test_flux_x(self):
        def test(x0, a, b, c, z0):
            v0, v1, v2 = np.array([
                    [x0, 0., 0.],
                    [x0 + a, 0., 0.],
                    [x0 + b, c, 0.]])
            
            target = np.array([0., 0., z0])
            normal = np.array([1., 0., 0.])
            
            correct = flux_exact_integrated(v0, v1, v2, target, [1., 0, 0])
            approx = B.flux_triangle(v0, v1, v2, target, normal)
            assert np.isclose(correct, approx, atol=0., rtol=5e-12), (x0, a,b,c, z0)

        N = 10
        for x0, a, b, c, z0 in zip(
                rand(N, min=0., max=100),
                rand(N, min=0., max=100),
                rand(N, min=-1000, max=1000),
                rand(N, min=-1000, max=1000),
                rand(N, min=-1, max=1)):

            test(x0, a, b, c, z0)

    def test_flux_x_special_case(self):
        def test(x0, a, b, c, z0):
            v0, v1, v2 = np.array([
                    [x0, 0., 0.],
                    [x0 + a, 0., 0.],
                    [x0 + b, c, 0.]])
            
            target = np.array([0., 0., z0])
            normal = np.array([1., 0., 0.])
            
            correct = flux_exact_integrated(v0, v1, v2, target, normal)
            approx = B.flux_triangle(v0, v1, v2, target, normal)
            assert np.isclose(correct, approx, atol=0., rtol=5e-12), (x0, a,b,c, z0)
         
        test(0.011464125694671257, 39.75569724092025, -752.9302350849487, -16.93841253122889, 0.3783263712792002)

    def test_pot_quadrants(self):
        z0 = 2
        
        for (a, b, c) in ([2., 2., 2.], [2., -2, 2]):
            v0, v1, v2 = np.array([
                [0., 0., 0.],
                [a, 0., 0.],
                [b, c, 0.]])
            
            target = np.array([0., 0., z0])
            area = np.linalg.norm(np.cross(v1-v0, v2-v0))/2
            correct = potential_exact_integrated(v0, v1, v2, target)
            approx = B.potential_triangle(v0, v1, v2, target)
            assert np.isclose(correct, approx)
    
    def test_flux_quadrants(self):
        z0 = 2
        
        for (a, b, c) in ([2., 2., 2.], [2., -2, 2]):
            v0, v1, v2 = np.array([
                [0., 0., 0.],
                [a, 0., 0.],
                [b, c, 0.]])
            
            normal = np.array([1., 1., 1.])
            target = np.array([0., 0., z0])
            
            area = np.linalg.norm(np.cross(v1-v0, v2-v0))/2
            correct = flux_exact_integrated(v0, v1, v2, target, normal)
            approx = B.flux_triangle(v0, v1, v2, target, normal)
            assert np.isclose(correct, approx)

    
    def test_pot_one_target_over_v0(self):
        target = np.array([0., 0., -5])
        v0, v1, v2 = np.array([
            [0., 0, 0],
            [3., 0., 0.],
            [-2., 7., 0.]])
            
        correct = potential_exact_integrated(v0, v1, v2, target)
        approx = B.potential_triangle(v0, v1, v2, target)
            
        assert np.isclose(correct, approx)
      
    def test_potential_singular(self):
        v0, v1, v2 = np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.]])

        target = np.array([2., 0., 0.])
        correct = potential_exact_integrated(v0, v1, v2, target) 
        approx = B.potential_triangle(v0, v1, v2, target)
        assert np.isclose(approx, correct)
        
        target = np.array([1. + 1e-5, 0., 0.])
        correct = potential_exact_integrated(v0, v1, v2, target) 
        approx = B.potential_triangle(v0, v1, v2, target)
        assert np.isclose(approx, correct)
        
        target = np.array([0., 1 + 1e-5, 0.])
        correct = potential_exact_integrated(v0, v1, v2, target) 
        approx = B.potential_triangle(v0, v1, v2, target)
        assert np.isclose(approx, correct)
        
        target = np.array([0., 0., 0.])
        correct = potential_exact_integrated(v0, v1, v2, target) 
        approx = B.potential_triangle(v0, v1, v2, target)
        assert np.isclose(approx, correct)
     
    def test_self_potential(self):
        for _ in range(10):
            v0, v1, v2 = rand(3,3)
            target = v0
             
            correct = potential_exact_integrated(v0, v1, v2, target) 
            approx = B.self_potential_triangle_v0(v0, v1, v2)
            assert np.allclose(correct, approx)
     
    def test_potential(self):
        for _ in range(10):
            v0, v1, v2 = rand(3,3)
            target = rand(3)
             
            correct = potential_exact_integrated(v0, v1, v2, target) 
            approx = B.potential_triangle(v0, v1, v2, target)
            assert np.allclose(correct, approx)

    def test_flux(self):
        for _ in range(10):
            v0, v1, v2 = rand(3,3)
            target = rand(3)
            normal = rand(3)
            
            correct = flux_exact_integrated(v0, v1, v2, target, normal) 
            approx = B.flux_triangle(v0, v1, v2, target, normal)
            assert np.allclose(correct, approx)
    
    def test_one_triangle_edwards(self):
        v0, v1, v2 = np.array([
            [11.591110,3.105829,5.000000],
 			[8.626804,3.649622,5.000000],
 			[9.000000,0.000000,5.000000]])
        
        target = np.array([9.739305,2.251817,5.000000])
        assert np.allclose(target, v0 + 1/3*(v1-v0) + 1/3*(v2-v0))
        assert np.isclose(
            potential_exact_integrated(v0, v1, v2, target), 
            B.potential_triangle(v0, v1, v2, target))
 

'''
class TestAdaptiveIntegration(unittest.TestCase):

    
    def test_x_squared(self):
        import time
        st = time.time()
        result = B.tanh_sinh_integration(lambda x: x*x, -1, 1, 1e-6, 1e-6)
        assert np.isclose(result, 2/3)
     

    def test_integration_ring(self):
        def pot(x):
            return B.potential_radial_ring(1.0, x, 1.0, 0.)
         
        correct = quad(pot, -0.1, 0., epsabs=1e-12, epsrel=1e-12)[0]
        ts = B.tanh_sinh_integration(pot, -0.1, 0., 1e-7, 1e-7)
        assert np.isclose(correct, ts)
        
        correct = quad(pot, 0., 0.1, epsabs=1e-12, epsrel=1e-12)[0]
        ts = B.tanh_sinh_integration(pot, 0., 0.1, 1e-7, 1e-7)
        assert np.isclose(correct, ts)
         
        def pot(x):
            return B.potential_radial_ring(1.0+x, 0+x, 1.0, 0.)
         
        correct = quad(pot, 0., 0.5, epsabs=1e-12, epsrel=1e-12)[0]
        ts = B.tanh_sinh_integration(pot, 0., 0.5, 1e-7, 1e-7)
        assert np.isclose(correct, ts)
        
        correct = quad(pot, 0., -0.5, epsabs=1e-12, epsrel=1e-12)[0]
        ts = B.tanh_sinh_integration(pot, 0., -0.5, 1e-7, 1e-7)
        assert np.isclose(correct, ts)
    
    def test_integration_ring2(self):
        def pot(x):
            return B.potential_radial_ring(10.0+x, 0, 10.0, 0.)
        correct = quad(pot, 0., 0.5, epsabs=1e-12, epsrel=1e-12)[0]
        ts = B.tanh_sinh_integration(pot, 0., 0.5, 1e-7, 1e-7)
        assert np.isclose(correct, ts)
    
    def test_integration_triangle(self):
        def singular_triangle(x):
            a, b, c = 0.02, 0., 2.0
            a, b, c = 10.000499987500625, -4.999750018748438, 0.049997500187484376
            z = 0.

            theta_min = 0.
            theta_max = atan2(c, b)
            
            theta = theta_min + (theta_max - theta_min) * (x + 1)/2
            
            rmax = -a*c/((b-a)*np.sin(theta)-c*np.cos(theta))
            return np.sqrt(z**2 + rmax**2) - abs(z)
        
        correct = quad(singular_triangle, -1., 1., epsabs=1e-12, epsrel=1e-12)[0]
        ts = B.tanh_sinh_integration(singular_triangle, -1., 1., 1e-5, 1e-5)

    def test_tanh_sinh_vs_quad(self):
        epsrel = 1e-8
        epsabs = 1e-8
        
        input_ = [ 
            (1., 2., lambda x: x*x),
            (-1, 5, lambda x: x + 5),
            (-10, 10, lambda x: sin(100*x)),
            (-10, 10, lambda x: 8*sin(5*x) * cos(x) + x**2 + atan(x)),
            (0, 1, lambda x: 1/sqrt(x))]
        
        for (x_min, x_max, integrand) in input_:
            correct = quad(integrand, x_min, x_max, epsabs=epsabs, epsrel=epsrel)[0]
            ts = B.tanh_sinh_integration(integrand, x_min, x_max, epsabs, epsrel)
            assert np.isclose(ts, correct, rtol=epsrel*10, atol=epsabs*10)

    def test_self_voltage_radial(self):
        return
        points = np.array([
            [0.026665, 0.599407],
            [0.035544, 0.598946]])
        r, z = (points[0] + points[1])/2.
        normal = points[1] - points[0]
        normal = np.array([-normal[1], normal[0]])
        normal /= np.linalg.norm(normal)
        K = 2.
        factor = 2*(K-1)/(1+K)
        
        def f(alpha):
            r0, z0 = points[0] + (points[1]-points[0])*alpha
            
            vec = np.array([
                B.dr1_potential_radial_ring(r0, z0, r, z),
                B.dz1_potential_radial_ring(r0, z0, r, z)])
            
            return factor * np.dot(-vec, normal)
        
        correct, _ = quad(f, 0, 0.5, epsabs=1e-7, epsrel=1e-7)
        ts = B.tanh_sinh_integration(f, 0, 0.5, 1e-5, 1e-5)
'''

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

    def test_magnetic_field_of_loop_against_backend2(self):
        N = 50
        test_cases = 10*np.random.rand(N, 4)
        
        for x0, y0, x, y in test_cases:
            traceon_field = mu_0*B.current_field_radial_ring(x0, y0, x, y)
            correct_field = magnetic_field_of_loop(1., x, np.array([x0, 0., y0-y]))
            
            assert np.isclose(traceon_field[0], correct_field[0], atol=1e-10)
            assert np.isclose(traceon_field[1], correct_field[2], atol=1e-10)
            
    
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
    
    def test_position_and_jacobian_radial(self):
        line = np.array([
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0]])

        _, pos = B.position_and_jacobian_radial(-1, *line)
        assert np.all(np.isclose(pos, line[0, :2]))
        _, pos = B.position_and_jacobian_radial(-1 + 2/3, *line)
        assert np.all(np.isclose(pos, line[1, :2]))
        _, pos = B.position_and_jacobian_radial(-1 + 4/3, *line)
        assert np.all(np.isclose(pos, line[2, :2]))
        _, pos = B.position_and_jacobian_radial(1, *line)
        assert np.all(np.isclose(pos, line[3, :2]))
    
    def test_position_and_jacobian_radial_length(self):
        line = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]])
        
        length = quad(lambda x: B.position_and_jacobian_radial(x, *line)[0], -1, 1)[0]
        assert np.isclose(length, 1.5)
        
        line = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0]])
        
        jac, pos = B.fill_jacobian_buffer_radial(np.array([line]))
        assert np.isclose(np.sum(jac[0]), 1.5)

        line = np.array([[ 2.1       , -0.74974852,  0.        ],
            [ 2.1       , -0.77751699,  0.        ],
            [ 2.1       , -0.80528545,  0.        ],
            [ 2.1       , -0.83305391,  0.        ]])
             
        length = quad(lambda x: B.position_and_jacobian_radial(x, *line)[0], -1, 1)[0]
        assert np.isclose(length, line[0, 1] - line[3, 1])

        normal = np.array([-1., 0., 0.])
        middle = np.mean(line, axis=0)

        def field_dot_normal_integrand(x):
            jac, pos = B.position_and_jacobian_radial(x, *line)
            
            field = np.array([
                -B.dr1_potential_radial_ring(middle[0], middle[1], pos[0], pos[1]),
                -B.dz1_potential_radial_ring(middle[0], middle[1], pos[0], pos[1]),
                0.0])

            return jac * np.dot(normal, field)
        
        field_dot_normal = quad(field_dot_normal_integrand, -1, 1, points=[-1., 0., 1.])[0]
        assert np.isclose(field_dot_normal, -0.01893954382812056)
    
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
            assert np.allclose(result, elec + mu_0*np.cross(vel, mag + current))
      
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
        tracer = T.Tracer(traceon_field, bounds, atol=1e-6)
        times, positions = tracer(initial_conditions[:3], T.velocity_vec(eV, [0, 0, -1]))
        
        interp = CubicSpline(positions[::-1, 2], np.array([positions[::-1, 0], positions[::-1, 1]]).T)
        
        assert np.allclose(interp(sol.y[2]), np.array([sol.y[0], sol.y[1]]).T)
          
    def test_current_field_ring(self):
        current = 2.5
        eff = get_ring_effective_point_charges(current, 1)
         
        for p in np.random.rand(10, 3):
            field = mu_0 * B.current_field(p, eff.charges, eff.jacobians, eff.positions)
            correct = biot_savart_loop(current, p)
            assert np.allclose(field, correct)
    
    def test_focus_2d(self):
        v1 = T.velocity_vec(10, [-1e-3, -1])
        v2 = T.velocity_vec(10, [1e-3, -1])

        p1 = np.concatenate( (3*v1, v1) )[np.newaxis, :]
        p2 = np.concatenate( (3*v2, v2) )[np.newaxis, :]

        (x, z) = F.focus_position([p1, p2])

        assert np.isclose(x, 0) and np.isclose(z, 0)

        p1[0, 0] += 1
        p2[0, 0] += 1

        (x, z) = F.focus_position([p1, p2])
        assert np.isclose(x, 1) and np.isclose(z, 0)

        p1[0, 1] += 1
        p2[0, 1] += 1

        (x, z) = F.focus_position([p1, p2])
        assert np.isclose(x, 1) and np.isclose(z, 1)

    def test_focus_3d(self):
        v1 = T.velocity_vec_spherical(1, 0, 0)
        v2 = T.velocity_vec_spherical(5, 1/30, 1/30)
        v3 = T.velocity_vec_spherical(10, 1/30, np.pi/2)

        p1 = np.concatenate( (v1, v1) )[np.newaxis, :]
        p2 = np.concatenate( (v2, v2) )[np.newaxis, :]
        p3 = np.concatenate( (v3, v3) )[np.newaxis, :]

        assert np.allclose(F.focus_position([p1, p2, p3]), [0., 0., 0.])





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
        eff = get_ring_effective_point_charges(current, 1.)
         
        traceon_field = S.FieldRadialBEM(current_point_charges=eff)
         
        z = np.linspace(-6, 6, 250)
        pot = [traceon_field.current_potential_axial(z_) for z_ in z]
        field = [traceon_field.current_field_at_point(np.array([0.0, z_]))[1] for z_ in z]

        numerical_derivative = CubicSpline(z, pot).derivative()(z)
        
        assert np.allclose(field, -numerical_derivative)
    
    def test_derivatives(self):
        current = 2.5
        eff = get_ring_effective_point_charges(current, 1.)
         
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
        eff = get_ring_effective_point_charges(current, 1.)
         
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
        traceon_field = S.FieldRadialBEM(current_point_charges=eff).axial_derivative_interpolation(-15, 15, N=500)
        tracer = T.Tracer(traceon_field, bounds, atol=1e-6)
        times, positions = tracer(initial_conditions[:3], T.velocity_vec(eV, [0, 0, -1]))
        
        interp = CubicSpline(positions[::-1, 2], np.array([positions[::-1, 0], positions[::-1, 1]]).T)
        
        assert np.allclose(interp(sol.y[2]), np.array([sol.y[0], sol.y[1]]).T, atol=1e-4, rtol=5e-5)

    def test_mag_pot_derivatives(self):
        with G.Geometry(G.Symmetry.RADIAL) as geom:
            points = [[0, 5], [5, 5], [5, -5], [0, -5]]
            lines = [geom.add_line(geom.add_point(p1), geom.add_point(p2)) for p1, p2 in zip(points, points[1:])]
            geom.add_physical(lines, 'boundary')
            
            r1 = geom.add_rectangle(1, 2, 2, 3, 0)
            r2 = geom.add_rectangle(1, 2, -3, -2, 0)
            
            geom.add_physical(r1.curves, 'r1')
            geom.add_physical(r2.curves, 'r2')
            geom.set_mesh_size_factor(10)
            mesh = geom.generate_line_mesh(False)
        
        e = E.Excitation(mesh)
        e.add_magnetostatic_potential(r1 = 10)
        e.add_magnetostatic_potential(r2 = -10)
         
        field = S.solve_bem(e)
        field_axial = field.axial_derivative_interpolation(-4.5, 4.5, N=1000)
          
        z = np.linspace(-4.5, 4.5, 300)
        derivs = field.get_magnetostatic_axial_potential_derivatives(z)
        z = z[5:-5]
           
        r = 0.3
        pot = np.array([field.potential_at_point(np.array([r, z_])) for z_ in z])
        pot_interp = np.array([field_axial.potential_at_point(np.array([r, z_])) for z_ in z])
        
        assert np.allclose(pot, pot_interp, rtol=1e-6)
          
        fr_direct = np.array([field.field_at_point(np.array([r, z_]))[0] for z_ in z])
        fz_direct = np.array([field.field_at_point(np.array([r, z_]))[1] for z_ in z])
        fr_interp = np.array([field_axial.field_at_point(np.array([r, z_]))[0] for z_ in z])
        fz_interp = np.array([field_axial.field_at_point(np.array([r, z_]))[1] for z_ in z])

        assert np.allclose(fr_direct, fr_interp, rtol=1e-3)
        assert np.allclose(fz_direct, fz_interp, rtol=1e-3)


class TestMagnetic(unittest.TestCase):

    def test_rectangular_coil(self):
        # Field produced by a 1mm x 1mm coil, with inner radius 2mm, 1ampere total current
        # What is the field produced at (2.5mm, 4mm)
        with G.Geometry(G.Symmetry.RADIAL) as geom:
            rect = geom.add_rectangle(2, 3, 2, 3, 0)
            geom.add_physical(rect.surface, 'coil')
            geom.set_mesh_size_factor(5)
            mesh = geom.generate_triangle_mesh(False)
        
        exc = E.Excitation(mesh)
        exc.add_current(coil=1)
        field = S.solve_bem(exc)

        assert np.isclose(np.sum(field.current_point_charges.jacobians), 1.0) # Area is 1.0
        assert np.isclose(np.sum(field.current_point_charges.charges[:, np.newaxis]*field.current_point_charges.jacobians), 1.0) # Total current is 1.0
        
        target = np.array([2.5, 0., 4.0])
        correct_r = dblquad(lambda x, y: magnetic_field_of_loop(1.0, x, np.array([target[0], 0.,target[2]-y]))[0], 2, 3, 2, 3)[0]
        correct_z = dblquad(lambda x, y: magnetic_field_of_loop(1.0, x, np.array([target[0], 0., target[2]-y]))[2], 2, 3, 2, 3)[0]
        
        computed = mu_0*field.current_field_at_point(np.array([target[0], target[2]]))
        correct = np.array([correct_r, correct_z])
        
        assert np.allclose(computed, correct, atol=1e-11)

if __name__ == '__main__':
    unittest.main()
















