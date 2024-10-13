import unittest, time
from math import *
import os.path as path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, dblquad
from scipy.constants import m_e, e, mu_0, epsilon_0

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


def get_ring_effective_point_charges(current, r):
    return S.EffectivePointCharges(
        [current],
        [ [1.] + ([0.]*(B.N_TRIANGLE_QUAD-1)) ],
        [ [[r, 0., 0.]] * B.N_TRIANGLE_QUAD ])

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

class TestBackend(unittest.TestCase):
    
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
















