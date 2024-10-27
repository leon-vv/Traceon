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
import traceon.backend as B
import traceon.logging as logging
from traceon.interpolation import FieldRadialAxial

from tests.test_radial_ring import potential_of_ring_arbitrary, biot_savart_loop, magnetic_field_of_loop

logging.set_log_level(logging.LogLevel.SILENT)

def get_ring_effective_point_charges(current, r):
    return S.EffectivePointCharges(
        [current],
        [ [1.] + ([0.]*(B.N_TRIANGLE_QUAD-1)) ],
        [ [[r, 0., 0.]] * B.N_TRIANGLE_QUAD ])

class TestRadial(unittest.TestCase):
    def test_charge_radial_vertical(self):
        vertices = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1/3],
            [1.0, 0.0, 2/3]])
        
        correct = 2*pi
        approx = B.charge_radial(vertices, 1.0);

        assert np.isclose(correct, approx)
    
    def test_charge_radial_horizontal(self):
        vertices = np.array([
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]])
         
        correct = 15*pi
        approx = B.charge_radial(vertices, 1.0);
         
        assert np.isclose(correct, approx)
    
    def test_charge_radial_skewed(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1/3, 0.0, 1/3],
            [2/3, 0.0, 2/3]])
        
        correct = pi*sqrt(2)
        approx = B.charge_radial(vertices, 1.0);

        assert np.isclose(correct, approx)

    def test_field_radial(self):
        vertices = np.array([
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [1.0 + 1/3, 0.0, 1.0 + 1/3],
            [1.0 + 2/3, 0.0, 1.0 + 2/3]])
        
        r0, z0 = 2.0, -2.5
        
        delta = 1e-5
        
        def Er(r, z):
            dVdr = (potential_of_ring_arbitrary(1.0, r0 + delta, z0, r, z)
                        - potential_of_ring_arbitrary(1.0,r0 - delta, z0, r, z))/(2*delta)
            return -dVdr
        
        def Ez(r, z):
            dVdz = (potential_of_ring_arbitrary(1.0, r0, z0 + delta, r, z)
                        - potential_of_ring_arbitrary(1.0, r0, z0 - delta, r, z))/(2*delta)
            return -dVdz
        
        length = sqrt(2)
        Er = quad(lambda x: Er(1.0 + x, 1.0 + x), 0.0, 1.0)[0] * length
        Ez = quad(lambda x: Ez(1.0 + x, 1.0 + x), 0.0, 1.0)[0] * length

        jac, pos = B.fill_jacobian_buffer_radial(np.array([vertices]))
        charges = np.ones(len(jac))
        assert np.allclose(B.field_radial(np.array([r0, z0]), charges, jac, pos)/epsilon_0, [Er, Ez], atol=0.0, rtol=1e-10)
     
    def test_rectangular_current_loop(self):
        coil = G.Surface.rectangle_xz(1.0, 2.0, 1.0, 2.0)
        coil.name = 'coil'

        mesh = coil.mesh(mesh_size=0.1)
         
        exc = E.Excitation(mesh, E.Symmetry.RADIAL)
        exc.add_current(coil=5)
        
        field = S.solve_direct(exc)
        
        z = np.linspace(-0.5, 3.5, 300)
        r = 0.0
        
        axial_field = np.array([field.current_field_at_point(np.array([r, z_]))[1] for z_ in z])
        
        file_ = path.join(path.dirname(__file__), 'axial magnetic field.txt')
        reference_data = np.loadtxt(file_, delimiter=',')
        reference = CubicSpline(reference_data[:, 0], reference_data[:, 1]*1e-3)
         
        assert np.allclose(reference(z), axial_field, atol=0., rtol=2e-2)
        
    def test_current_field_ring(self):
        current = 2.5
        eff = get_ring_effective_point_charges(current, 1)
         
        for p in np.random.rand(10, 3):
            field = mu_0 * B.current_field(p, eff.charges, eff.jacobians, eff.positions)
            correct = biot_savart_loop(current, p)
            assert np.allclose(field, correct)
    
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
            assert np.allclose(deriv, derivatives[:, i+1], atol=1e-5, rtol=5e-3)       
    
    def test_interpolation_current_loop(self):
        current = 2.5
        eff = get_ring_effective_point_charges(current, 1.)
         
        traceon_field = S.FieldRadialBEM(current_point_charges=eff)

        interp = FieldRadialAxial(traceon_field, -5, 5, N=300)
         
        z = interp.z[1:-1]

        pot = [traceon_field.current_potential_axial(z_) for z_ in z]
        pot_interp = [interp.magnetostatic_potential_at_point(np.array([0., z_])) for z_ in z]

        assert np.allclose(pot, pot_interp)
         
        r = np.linspace(-0.5, 0.5, 5)
        
        for r_ in r:
            field = [traceon_field.magnetostatic_field_at_point(np.array([r_, z_]))[1] for z_ in z]
            field_interp = [interp.magnetostatic_field_at_point(np.array([r_, z_]))[1] for z_ in z]
            assert np.allclose(field, field_interp, atol=1e-3, rtol=5e-3)
     
    def test_mag_pot_derivatives(self):
        boundary = G.Path.line([0., 0., 5.], [5., 0., 5.])\
            .line_to([5., 0., -5.])\
            .line_to([0., 0., -5.])
        
        r1 = G.Path.rectangle_xz(1, 2, 2, 3)
        r2 = G.Path.rectangle_xz(1, 2, -3, -2)

        boundary.name = 'boundary'
        r1.name = 'r1'
        r2.name = 'r2'
        
        mesh = (boundary + r1 + r2).mesh(mesh_size=0.1)
         
        e = E.Excitation(mesh, E.Symmetry.RADIAL)
        e.add_magnetostatic_potential(r1 = 10)
        e.add_magnetostatic_potential(r2 = -10)
         
        field = S.solve_direct(e)
        field_axial = FieldRadialAxial(field, -4.5, 4.5, N=1000)
          
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
    
    def test_rectangular_coil(self):
        # Field produced by a 1mm x 1mm coil, with inner radius 2mm, 1ampere total current
        # What is the field produced at (2.5mm, 4mm)
        coil = G.Surface.rectangle_xz(2, 3, 2, 3)
        coil.name = 'coil'
        
        mesh = coil.mesh(mesh_size=0.1)
        
        exc = E.Excitation(mesh, E.Symmetry.RADIAL)
        exc.add_current(coil=1)
        field = S.solve_direct(exc)

        assert np.isclose(np.sum(field.current_point_charges.jacobians), 1.0) # Area is 1.0
        assert np.isclose(np.sum(field.current_point_charges.charges[:, np.newaxis]*field.current_point_charges.jacobians), 1.0) # Total current is 1.0
        
        target = np.array([2.5, 0., 4.0])
        computed = mu_0*field.current_field_at_point(np.array([target[0], target[2]]))
        
        correct_r = dblquad(lambda x, y: mu_0*B.current_field_radial_ring(target[0], target[2], x, y)[0], 2, 3, 2, 3, epsrel=1e-4)[0]
        correct_z = dblquad(lambda x, y: mu_0*B.current_field_radial_ring(target[0], target[2], x, y)[1], 2, 3, 2, 3, epsrel=1e-4)[0]
        correct = np.array([correct_r, correct_z])

        assert np.allclose(computed, correct, atol=0.0, rtol=1e-9) 

