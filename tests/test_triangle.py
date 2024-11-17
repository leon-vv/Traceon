import unittest

from scipy.integrate import dblquad
import numpy as np

import traceon.backend as B

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
    return dblquad(potential_exact, 0, 1, 0, lambda x: 1-x, epsabs=1e-10, epsrel=1e-10, args=(target, (v0, v1, v2)))[0] * (2*area)

def flux_exact_integrated(v0, v1, v2, target, normal, epsabs=1e-10, epsrel=1e-10):
    area = np.linalg.norm(np.cross(v1-v0, v2-v0))/2
    return dblquad(flux_exact, 0, 1, 0, lambda x: 1-x, epsabs=epsabs, epsrel=epsrel, args=(target, (v0, v1, v2), normal))[0] * (2*area)


class TestTriangle(unittest.TestCase):

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
            assert np.isclose(approx, correct, atol=0., rtol=5e-10), (a,b,c)

        for (a,b,c) in rand(3, 3):
            test(a,b,c)
    
    def test_potential_triangle_close(self):
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        range_ = [-20, -16, -4, -3, -2, -1, 0, 1, 2, 3, 4, 8, 20]

        for x in range_:
            target = np.array([x, 0.0, 0.5])
            assert np.isclose(B.potential_triangle(v0, v1, v2, target), potential_exact_integrated(v0, v1, v2, target), atol=0.0, rtol=1e-9)

        for x in range_:
            target = np.array([x, 0.0, -0.5])
            assert np.isclose(B.potential_triangle(v0, v1, v2, target), potential_exact_integrated(v0, v1, v2, target), atol=0.0, rtol=1e-9)

        for y in range_:
            target = np.array([0.0, y, 0.5])
            assert np.isclose(B.potential_triangle(v0, v1, v2, target), potential_exact_integrated(v0, v1, v2, target), atol=0.0, rtol=1e-9)

        for y in range_:
            target = np.array([0.0, y, -0.5])
            assert np.isclose(B.potential_triangle(v0, v1, v2, target), potential_exact_integrated(v0, v1, v2, target), atol=0.0, rtol=1e-9)

        for z in range_:
            target = np.array([1.0, 1.0, z])
            assert np.isclose(B.potential_triangle(v0, v1, v2, target), potential_exact_integrated(v0, v1, v2, target), atol=0.0, rtol=1e-9)

        for z in range_:
            target = np.array([-1.0, -1.0, z])
            assert np.isclose(B.potential_triangle(v0, v1, v2, target), potential_exact_integrated(v0, v1, v2, target), atol=0.0, rtol=1e-9)

        for z in range_:
            target = np.array([-0.5, 0.5, z])
            assert np.isclose(B.potential_triangle(v0, v1, v2, target), potential_exact_integrated(v0, v1, v2, target), atol=0.0, rtol=1e-9)

        for z in range_:
            target = np.array([0.5, -0.5, z])
            assert np.isclose(B.potential_triangle(v0, v1, v2, target), potential_exact_integrated(v0, v1, v2, target), atol=0.0, rtol=1e-9)

        for k in range_:
            target = np.array([k, k, 0.5])
            assert np.isclose(B.potential_triangle(v0, v1, v2, target), potential_exact_integrated(v0, v1, v2, target), atol=0.0, rtol=1e-9)
    
    def test_potential_triangle_edge_cases(self):
        edge_cases = [
            # Note that the target lies in the same plane
            # and in such a way that it's between the y-axis limits of the triangle    
            (-3, -0.5, 2.0, 1.0, 1.0, 0.0),
            # Note that the target lies in the same plane
            # and in such a way that it's between the y-axis limits of the triangle    
            (-3, -0.5, 2.0, 1.0, 1.0, 0.001),
            
            # Target point just outside the left edge of the triangle 
            (-1, -0.5, 4.0, 4.0, 1.5, 0.0),
            # Target point just outside the right edge of the triangle 
            (-2, -0.5, 2.5, 1.0, 1.0, 0.0),
            # Target point 'under' the triangle at the right side
            (-3, -0.5, 1, 4.25, 1.0, 0.0),
            # Target point 'under' the triangle at the left side
            (1.0, -0.5, 1.0, -2.5, 2.0, 0.0)]
        
        for (x0, y0, a, b, c, z) in edge_cases:
            v0, v1, v2 = np.array([
                [x0, y0, 0.],
                [x0 + a, y0, 0.],
                [x0 + b, y0 + c, 0.]])
            
            target = np.array([0., 0., z])
            correct = potential_exact_integrated(v0, v1, v2, target)
            approx = B.potential_triangle(v0, v1, v2, target)
            assert np.isclose(correct, approx)


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
            
            correct = flux_exact_integrated(v0, v1, v2, target, [1., 0, 0], epsabs=0.0, epsrel=5e-10)
            approx = B.flux_triangle(v0, v1, v2, target, normal)
            assert np.isclose(correct, approx, atol=0., rtol=2e-9), (x0, a,b,c, z0)

        possible_x0 = [-10, -1, -0.1, 0.1, 1, 10]
        possible_a = [0.1, 5, 10, 25]
        possible_b = [-10, -1, -0.1, 0.1, 1, 10]
        possible_c = [-10, -1, -0.1, 0.1, 1, 10]
        possible_z0 = [-10, -1, -0.1, 0.1, 1, 10]
        
        for _ in range(10):
            x0 = np.random.choice(possible_x0)
            a = np.random.choice(possible_a)
            b = np.random.choice(possible_b)
            c = np.random.choice(possible_c)
            z0 = np.random.choice(possible_z0)
            test(x0, a, b, c, z0)
    
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
    
    def test_flux_triangle_edge_cases(self):
        edge_cases = [
            # Note that the target lies in the same plane
            # and in such a way that it's between the y-axis limits of the triangle    
            (-3, -0.5, 2.0, 1.0, 1.0, 0.0),
            # Note that the target lies in the same plane
            # and in such a way that it's between the y-axis limits of the triangle    
            (-3, -0.5, 2.0, 1.0, 1.0, 0.001),
            
            # Target point just outside the left edge of the triangle 
            (-1, -0.5, 4.0, 4.0, 1.5, 0.0),
            # Target point just outside the right edge of the triangle 
            (-2, -0.5, 2.5, 1.0, 1.0, 0.0),
            # Target point 'under' the triangle at the right side
            (-3, -0.5, 1, 4.25, 1.0, 0.0),
            # Target point 'under' the triangle at the left side
            (1.0, -0.5, 1.0, -2.5, 2.0, 0.0)]
        
        normal = np.array([1., 1., 1.])
    
        for (x0, y0, a, b, c, z) in edge_cases:
            v0, v1, v2 = np.array([
                [x0, y0, 0.],
                [x0 + a, y0, 0.],
                [x0 + b, y0 + c, 0.]])
            
            target = np.array([0., 0., z])
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
        for _ in range(3):
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
    
    def test_flux_triangle_close(self):
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        normals = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        
        range_ = [-20, -16, -4, -3, -2, -1, 0, 1, 2, 3, 4, 16, 20]
        
        for normal in normals:
            for x in range_:
                target = np.array([x, 0.0, 0.5])
                assert np.isclose(B.flux_triangle(v0, v1, v2, target, normal), flux_exact_integrated(v0, v1, v2, target, normal), atol=0.0, rtol=1e-8)

            for x in range_:
                target = np.array([x, 0.0, -0.5])
                assert np.isclose(B.flux_triangle(v0, v1, v2, target, normal), flux_exact_integrated(v0, v1, v2, target, normal), atol=0.0, rtol=1e-8)

            for y in range_:
                target = np.array([0.0, y, 0.5])
                assert np.isclose(B.flux_triangle(v0, v1, v2, target, normal), flux_exact_integrated(v0, v1, v2, target, normal), atol=0.0, rtol=1e-8)

            for y in range_:
                target = np.array([0.0, y, -0.5])
                assert np.isclose(B.flux_triangle(v0, v1, v2, target, normal), flux_exact_integrated(v0, v1, v2, target, normal), atol=0.0, rtol=1e-8)

            for z in range_:
                target = np.array([1.0, 1.0, z])
                assert np.isclose(B.flux_triangle(v0, v1, v2, target, normal), flux_exact_integrated(v0, v1, v2, target, normal), atol=0.0, rtol=1e-8)

            for z in range_:
                target = np.array([-1.0, -1.0, z])
                assert np.isclose(B.flux_triangle(v0, v1, v2, target, normal), flux_exact_integrated(v0, v1, v2, target, normal), atol=0.0, rtol=1e-8)

            for z in range_:
                target = np.array([-0.5, 0.5, z])
                assert np.isclose(B.flux_triangle(v0, v1, v2, target, normal), flux_exact_integrated(v0, v1, v2, target, normal), atol=0.0, rtol=1e-8)

            for z in range_:
                target = np.array([0.5, -0.5, z])
                assert np.isclose(B.flux_triangle(v0, v1, v2, target, normal), flux_exact_integrated(v0, v1, v2, target, normal), atol=0.0, rtol=1e-8)

            for k in range_:
                target = np.array([k, k, 0.5])
                assert np.isclose(B.flux_triangle(v0, v1, v2, target, normal), flux_exact_integrated(v0, v1, v2, target, normal), atol=0.0, rtol=1e-8)

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
 
