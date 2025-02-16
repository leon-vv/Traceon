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
from traceon.field import FieldRadialAxial

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
        assert np.allclose(B.field_radial(np.array([r0, 0.0, z0]), charges, jac, pos)/epsilon_0, [Er, 0.0, Ez], atol=0.0, rtol=1e-10)
     
    def test_rectangular_current_loop(self):
        coil = G.Surface.rectangle_xz(1.0, 2.0, 1.0, 2.0)
        coil.name = 'coil'

        mesh = coil.mesh(mesh_size=0.1)
         
        exc = E.Excitation(mesh, E.Symmetry.RADIAL)
        exc.add_current(coil=5)
        
        field = S.solve_direct(exc)
        
        z = np.linspace(-0.5, 3.5, 300)
        r = 0.0
        
        axial_field = np.array([field.current_field_at_point([r, 0.0, z_])[2] for z_ in z])
        
        file_ = path.join(path.dirname(__file__), 'axial magnetic field.txt')
        reference_data = np.loadtxt(file_, delimiter=',')
        reference = CubicSpline(reference_data[:, 0], reference_data[:, 1]*1e-3)
         
        assert np.allclose(reference(z), axial_field, atol=0., rtol=2e-2)
        
    def test_current_field_ring(self):
        current = 2.5
        eff = get_ring_effective_point_charges(current, 1)
         
        for p in np.random.rand(10, 3):
            field = mu_0 * B.current_field_radial(p, eff.charges, eff.jacobians, eff.positions)
            correct = biot_savart_loop(current, p)
            assert np.allclose(field, correct)
    
    def test_current_loop(self):
        current = 2.5
        eff = get_ring_effective_point_charges(current, 1.)
         
        traceon_field = S.FieldRadialBEM(current_point_charges=eff)
         
        z = np.linspace(-6, 6, 250)
        pot = [traceon_field.current_potential_axial(z_) for z_ in z]
        field = [traceon_field.current_field_at_point([0.0, 0.0, z_])[2] for z_ in z]

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
        pot_interp = [interp.magnetostatic_potential_at_point([0., 0.0, z_]) for z_ in z]

        assert np.allclose(pot, pot_interp)
         
        r = np.linspace(-0.5, 0.5, 5)
        
        for r_ in r:
            field = [traceon_field.magnetostatic_field_at_point([r_, 0.0, z_])[1] for z_ in z]
            field_interp = [interp.magnetostatic_field_at_point([r_, 0.0, z_])[1] for z_ in z]
            assert np.allclose(field, field_interp, atol=1e-3, rtol=5e-3)
     
    def test_mag_pot_derivatives(self):
        boundary = G.Path.line([0., 0., 5.], [5., 0., 5.])\
            .extend_with_line([5., 0., -5.])\
            .extend_with_line([0., 0., -5.])
        
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
        pot = [field.potential_at_point([r, 0.0, z_]) for z_ in z]
        pot_interp = [field_axial.potential_at_point([r, 0.0, z_]) for z_ in z]
        
        assert np.allclose(pot, pot_interp, rtol=1e-6)
          
        field_direct = [field.field_at_point([r, 0.0, z_]) for z_ in z]
        field_interp = [field_axial.field_at_point([r, 0.0, z_]) for z_ in z]
        assert np.allclose(field_direct, field_interp, rtol=1e-3)
    
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
        computed = mu_0*field.current_field_at_point(target)
        
        correct_r = dblquad(lambda x, y: mu_0*B.current_field_radial_ring(target[0], target[2], x, y)[0], 2, 3, 2, 3, epsrel=1e-4)[0]
        correct_z = dblquad(lambda x, y: mu_0*B.current_field_radial_ring(target[0], target[2], x, y)[1], 2, 3, 2, 3, epsrel=1e-4)[0]
        correct = np.array([correct_r, 0.0, correct_z])

        assert np.allclose(computed, correct, atol=0.0, rtol=1e-9) 

    def test_field_superposition(self):
        boundary = G.Path.line([0., 0., 0.], [5., 0., 0.]).extend_with_line([5., 0., 5.]).extend_with_line([0., 0., 5.])
        boundary.name = 'boundary'

        rect1 = G.Path.rectangle_xz(1.0, 2.0, 1.0, 2.0)
        rect1.name = 'rect1'

        rect2 = G.Path.rectangle_xz(1.0, 2.0, 3.0, 4.0)
        rect2.name = 'rect2'

        rect3 = G.Surface.rectangle_xz(3.0, 4.0, 3.0, 4.0)
        rect3.name = 'rect3'

        rect4 = G.Path.rectangle_xz(3.0, 4.0, 1.0, 2.0)
        rect4.name = 'rect4'

        mesh = (boundary + rect1 + rect2 + rect4).mesh(mesh_size_factor=5) + rect3.mesh(mesh_size_factor=2)

        exc = E.Excitation(mesh, E.Symmetry.RADIAL)
        exc.add_magnetostatic_boundary('boundary')
        exc.add_voltage(rect1=10)
        exc.add_voltage(rect2=0.0)
        exc.add_current(rect3=2.5)
        exc.add_dielectric(rect4=8)

        field = S.solve_direct(exc)

        # Excitation with half the values
        exc_half = E.Excitation(mesh, E.Symmetry.RADIAL)
        exc_half.add_magnetostatic_boundary('boundary')
        exc_half.add_voltage(rect1=10 / 2.)
        exc_half.add_voltage(rect2=1.0)
        exc_half.add_current(rect3=2.5 / 2.)
        exc_half.add_dielectric(rect4=8)

        superposition = S.solve_direct_superposition(exc_half)

        # The following expression is written in a weird way to test many corner cases
        numpy_minus_two = np.array([-2.0])[0] # Numpy scalars sometimes give issues
        superposed = -2.0* (-superposition['rect1']) - superposition['rect3']*numpy_minus_two + 0*superposition['rect2']

        # Field and superposed should be EXACTLY the same!
        for (eff1, eff2) in zip([field.electrostatic_point_charges, field.magnetostatic_point_charges, field.current_point_charges],
                                [superposed.electrostatic_point_charges, superposed.magnetostatic_point_charges, superposed.current_point_charges]):
            assert np.allclose(eff1.jacobians, eff2.jacobians)
            assert np.allclose(eff1.charges, eff2.charges)
            assert np.allclose(eff1.positions, eff2.positions)
            assert eff1.directions == eff2.directions or np.allclose(eff1.directions, eff2.directions)

        # Since the fields are the same they should return the same values at some arbitrary points
        points = [ [0.5, 0.5, 0.5], [1.0, 0.0, 2.0], [2.0, 0.0, -2.0] ]

        for p in points:
            assert np.allclose(field.electrostatic_field_at_point(p), superposed.electrostatic_field_at_point(p))
            assert np.allclose(field.magnetostatic_field_at_point(p), superposed.magnetostatic_field_at_point(p))
            assert np.allclose(field.current_field_at_point(p), superposed.current_field_at_point(p))


class TestRadialPermanentMagnet(unittest.TestCase):
    def test_triangular_permanent_magnet(self):
        triangle = G.Path.line([0.5, 0., -0.25], [0.5, 0., 0.25])\
            .extend_with_line([1., 0., 0.25]).close()
        triangle.name = 'triangle'
        
        mesh = triangle.mesh(mesh_size_factor=15, higher_order=True)
        
        exc = E.Excitation(mesh, E.Symmetry.RADIAL)
        exc.add_permanent_magnet(triangle=np.array([0., 0., 2.]))
        
        solver = S.MagnetostaticSolverRadial(exc)
        field = solver.get_permanent_magnet_field()
        
        evaluation_points = np.array([
            [0.8, 0.],
            [1.0, 0.],
            [0.75, 0.2],
            [0.75, 0.5]])

        # Taken from a FEM package
        comparison = np.array([
            [0.28152, 0.098792, -0.47191, -0.19345],
            [-1.3549, 0.18957, -0.020779, -0.12875]]).T

        comparison = np.array([
            [-0.47191, -0.020779],
            [-0.19345, -0.12875],
            [0.28152, -1.3549],
            [0.098792, 0.18957]
        ])

        for ev, comp in zip(evaluation_points, comparison):
            Hr, _, Hz = field.magnetostatic_field_at_point([ev[0], 0., ev[1]])
            assert np.allclose([mu_0*Hr, mu_0*Hz], comp, rtol=1e-5, atol=0.004)

    def test_field_along_axis_of_cylindrical_permanent_magnet(self):
        R = 1.5 # Radius of magnet
        Br = 2.0 # Remanent flux of magnet
        L = 3 # Length of magnet

        rect = G.Path.line([0., 0., 0.], [R, 0., 0.])\
            .extend_with_line([R, 0., L]).extend_with_line([0., 0., L])  
        rect.name = 'magnet'

        mesh = rect.mesh(mesh_size_factor=5)
         
        e = E.Excitation(mesh, E.Symmetry.RADIAL)
        e.add_permanent_magnet(magnet=[0., 0., Br])
        
        field = S.solve_direct(e)
         
        for dz in [0.1, 1, 5, 10]:
            # See https://e-magnetica.pl/doku.php/calculator/magnet_cylindrical
            correct = Br / (2*mu_0) * ( (dz+L)/sqrt(R**2 + (dz+L)**2) - dz/sqrt(R**2 + dz**2))
            
            Hz = field.magnetostatic_field_at_point([0., 0., L+dz])[2]
            assert np.isclose(correct, Hz)

    def test_magnet_with_magnetizable_material(self):
        magnet = G.Path.circle_xz(3, -2, 1)
        magnet.name = 'magnet'

        magnetizable = G.Path.circle_xz(3, 2, 1)
        magnetizable.name = 'circle'

        mesh = (magnet + magnetizable).mesh(mesh_size_factor=40, higher_order=True)

        exc = E.Excitation(mesh, E.Symmetry.RADIAL)
        exc.add_permanent_magnet(magnet=[0, 0, 2.])
        exc.add_magnetizable(circle=5)

        field = S.solve_direct(exc)

        points = np.array([
            [3, -4],
            [3, -2],
            [3, -0.5],
            [3, 0.5],
            [3, 2],
            [3, 4]])

        # From a FEM package
        comparison = np.array([
            [-0.080149, 0.25363],
            [-0.003367, -1.0145],
            [0.085632, 0.45160],
            [0.046463, 0.19612],
            [0.012013, 0.023293],
            [0.015441, 0.042159]])

        for (c, (r, z)) in zip(comparison, points):
            Hr, _, Hz = field.magnetostatic_field_at_point([r, 0., z])
            assert np.isclose(c[0], mu_0*Hr, rtol=1e-4, atol=0.0002)
            assert np.isclose(c[1], mu_0*Hz, rtol=1e-4, atol=0.0002)
 


class TestSimpleMagneticLens(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        coil = G.Surface.rectangle_xz(3, 4, 2, 3)
        coil.name = 'coil'

        pole = G.Path.line([1, 0, 0], [4, 0, 0])\
            .extend_with_line([4, 0, 1])\
            .extend_with_line([2, 0, 1])\
            .extend_with_line([2, 0, 3])\
            .extend_with_line([1, 0, 3])\
            .close()
        pole.name = 'pole'
        
        MSF = 10
        mesh = coil.mesh(mesh_size_factor=20) + pole.mesh(mesh_size_factor=MSF)
        
        MU = 100

        exc = E.Excitation(mesh, E.Symmetry.RADIAL)
        exc.add_current(coil=1)
        exc.add_magnetizable(pole=MU)

        cls.field = S.solve_direct(exc)
        cls.field_axial = FieldRadialAxial(cls.field, -10, 10, N=500)

    def test_potential_is_zeroth_derivative(self):
        z = np.linspace(-8, 8, 100)
        mag_pot = np.array([self.field.magnetostatic_potential_at_point([0.0, 0.0, z_]) for z_ in z])
        current_pot = np.array([self.field.current_potential_axial(z_) for z_ in z])
        
        derivatives = self.field.get_magnetostatic_axial_potential_derivatives(z)

        assert np.allclose(current_pot + mag_pot, derivatives[:, 0])

    def test_Hz_is_first_derivative(self):
        z = np.linspace(-8, 8, 100)
        Hz = np.array([self.field.field_at_point([0., 0., z_])[2] for z_ in z])
        derivatives = self.field.get_magnetostatic_axial_potential_derivatives(z)

        assert np.allclose(-derivatives[:, 1], Hz)

    def test_field_and_axial_agree(self):
        z = np.linspace(-8, 8, 100)

        field_direct = np.array([self.field.field_at_point([0.25, 0., z_]) for z_ in z])
        field_axial = np.array([self.field_axial.field_at_point([0.25, 0., z_]) for z_ in z])
        assert np.allclose(field_direct, field_axial, rtol=5e-5)
    
    def test_field_equals_comsol(self):
        comsol = [-0.0081196, 0., 0.024124]
        f = self.field.field_at_point([0.4, 0., 2.])
        # Increasing MSF gets this tolerance to below 5%
        # Also there is some discrepancy with the Comsol model,
        # as the Comsol model includes a boundary
        assert np.allclose(comsol, f, rtol=30e-2)

class TestFlatEinzelLens(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        boundary = G.Path.line([0.1, 0.0, 1.0], [1.0, 0.0, 1.0])\
            .extend_with_line([1.0, 0.0, -1.0]).extend_with_line([0.1, 0.0, -1.0])
        
        ground_top = G.Path.line([0.25, 0.0, 0.5], [0.75, 0.0, 0.5])
        lens = G.Path.line([0.25, 0.0, 0.0], [0.75, 0.0, 0.0])
        ground_bottom = G.Path.line([0.25, 0.0, -0.5], [0.75, 0.0, -0.5])

        boundary.name = 'boundary'
        ground_top.name = 'ground'
        lens.name = 'lens'
        ground_bottom.name = 'ground'

        mesh = (boundary + ground_top + lens + ground_bottom).mesh(mesh_size_factor=40)

        exc = E.Excitation(mesh, E.Symmetry.RADIAL)

        exc.add_voltage(ground=0, lens=1000)
        exc.add_electrostatic_boundary('boundary')
         
        cls.z = np.linspace(-0.85, 0.85, 250)
        cls.field = S.solve_direct(exc)
        cls.field_axial = FieldRadialAxial(cls.field, cls.z[0], cls.z[-1], N=500)
    
    def test_derivatives(self):
        derivatives = self.field.get_electrostatic_axial_potential_derivatives(self.z)
        derivatives_spline = [CubicSpline(self.z, derivatives[:, i])(self.z, nu=1) for i in range(derivatives.shape[1])]
        
        for i in range(6):
            assert np.allclose(derivatives[:, i+1], derivatives_spline[i], atol=1e-12, rtol=1e-2)

    def test_potential_close_to_axis(self):
        r = 0.1
        pot = [self.field.potential_at_point([r, 0.0, z_]) for z_ in self.z]
        pot_axial = [self.field_axial.potential_at_point([r, 0.0, z_]) for z_ in self.z]
        assert np.allclose(pot_axial[2:-2], pot[2:-2], atol=0.0, rtol=1e-5)
    
    def test_field_close_to_axis_xz(self):
        r = 0.05 # In XZ plane
        f = [self.field.field_at_point([r, 0.0, z_]) for z_ in self.z[1:-1]]
        f_axial = [self.field_axial.field_at_point([r, 0.0, z_]) for z_ in self.z[1:-1]]
        assert np.allclose(f, f_axial, atol=1e-10, rtol=1e-5)
    
    def test_field_close_to_axis_xy(self):
        r = 0.05 # In YZ plane
        f = [self.field.field_at_point([0.0, r, z_]) for z_ in self.z[1:-1]]
        f_axial = [self.field_axial.field_at_point([0.0, r, z_]) for z_ in self.z[1:-1]]
        assert np.allclose(f, f_axial, atol=1e-10, rtol=1e-5)

    def test_trace_close_to_axis(self):
        r = 0.05
        z = 0.85

        tracer = self.field.get_tracer( [(-0.1, 0.1), (-0.1, 0.1), (-0.85, 0.85)] )
        tracer_axial = self.field_axial.get_tracer( [(-0.1, 0.1), (-0.1, 0.1), (-0.85, 0.85)] )
        
        p0 = np.array([r, 0.0, z])
        v0 = T.velocity_vec(100, [0, 0, -1])

        _, pos = tracer(p0, v0)
        _, pos_axial = tracer_axial(p0, v0)

        intersection = T.xy_plane_intersection(pos, -0.8)
        intersection_axial = T.xy_plane_intersection(pos_axial, -0.8)
         
        assert np.allclose(intersection, intersection_axial, rtol=5e-4)

















