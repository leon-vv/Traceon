from math import sqrt, cos, sin
import time

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy
from scipy.integrate import *

EM = -0.1758820022723908 # e/m units ns and mm

def velocity_vec(eV, theta, direction=1):
    """Compute a velocity vector (np.array of shape (2,)).
    
    Args:
        eV: the energy of the electron
        theta: angle with respect to the optical axis (r=0)
        direction: whether the vector points in the positive z direction
            (direction > 0) or in the negative z direction (direction < 0)
    
    Returns:
        Velocity vector (np.array of shape (2,))
    """

    # From electronvolt to mm/ns
    V = 0.5930969604919433*sqrt(eV)
    factor = 1.0 if direction > 0 else -1.0
    return np.array([V*sin(theta), factor*V*cos(theta)])

def _angle(vr, vz):
    return np.sign(vr) * np.arctan(np.abs(vr/vz))

STEP_MAX = 0.085
STEP_MIN = STEP_MAX/1e10

@nb.njit
def trace_particle(position, velocity, field, rmax, zmin, zmax, rmin=None, args=(), atol=1e-10):
    """Trace a particle. Using the Runge-Kutta-Fehlberg method RK45. See:
        
        https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method

        Erwin Fehlberg. Low-Order Classical Runge-Kutta Formulas With Stepsize Control and their Application to Some Heat
        Transfer Problems. 1969. National Aeronautics and Space Administration.
    
    Args:
        position: starting position of the particle
        velocity: starting velocity vector of the particle (see 'velocity_vec')
        field: field function (see solver.py)
        rmax: maximum r value allowed, when a particle goes outside [-rmax,rmax]
            the tracing will end
        rmin: optional, minimum r value allowed, when a particle goes outside [rmin,rmax]
            the tracing will end
        zmin: minimum value of z
        zmax: maximum value of z, when a particle goes outside the bounds [zmin,zmax]
            the tracing will end
        args: extra arguments passed to field, besides r and z. Useful to supply voltages
            when the field function is a result of a superposition (see solver.py)
    
    Returns:
        np.narray of shape (N, 4) where N is the number of time steps taken. 
    """
    Nmax = 50000
    V = np.linalg.norm(velocity)
    h = STEP_MAX/V
    hmax = STEP_MAX/V
    hmin = STEP_MIN/V
    
    def f(_, y):
        Er, Ez = field(y[0], y[1], *args)
        return np.array([y[2], y[3], EM*Er, EM*Ez])
       
    y = np.array([position[0], position[1], velocity[0], velocity[1]])
    
    N = 1
    positions = np.zeros( (Nmax, 4) )
    positions[0, :] = y

    A = (0.0, 0.0, 2/9, 1/3, 3/4, 1, 5/6) # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    B6 = (0.0, 65/432, -5/16, 13/16, 4/27, 5/144) # Left pad with 0.0 to keep indices the same
    B5 = (0.0, -17/12, 27/4, -27/5, 16/15)
    B4 = (0.0, 69/128, -243/128, 135/64)
    B3 = (0.0, 1/12, 1/4)
    B2 = (0.0, 2/9)
    CH = (0.0, 47/450, 0, 12/25, 32/225, 1/30, 6/25)
    CT = (0.0, -1/150, 0, 3/100, -16/75, -1/20, 6/25)

    rmin = -rmax if rmin is None else rmin
     
    while rmin <= y[0] <= rmax and zmin <= y[1] <= zmax:
        k1 = h * f(0.0, y)
        k2 = h * f(0.0, y + B2[1]*k1)
        k3 = h * f(0.0, y + B3[1]*k1 + B3[2]*k2)
        k4 = h * f(0.0, y + B4[1]*k1 + B4[2]*k2 + B4[3]*k3)
        k5 = h * f(0.0, y + B5[1]*k1 + B5[2]*k2 + B5[3]*k3 + B5[4]*k4)
        k6 = h * f(0.0, y + B6[1]*k1 + B6[2]*k2 + B6[3]*k3 + B6[4]*k4 + B6[5]*k5)
        
        TE = np.max(np.abs(CT[1]*k1 + CT[2]*k2 + CT[3]*k3 + CT[4]*k4 + CT[5]*k5 + CT[6]*k6))
         
        if TE <= atol or h == hmin:
            y = y + CH[1]*k1 + CH[2]*k2 + CH[3]*k3 + CH[4]*k4 + CH[5]*k5 + CH[6]*k6
            assert N < Nmax
            positions[N, :] = y
            N += 1
         
        if TE > 1e-11: 
            h = max(min(0.9*h*(atol/TE)**(1/5), hmax), hmin)
        elif TE < 1e-11:
            h = hmax
     
    return positions[:N]

@nb.njit(cache=True)
def trace_particle_rk4(position, velocity, field, rmax, zmin, zmax, rmin=None, args=(), mm_per_step=0.015):
    """Trace a particle using the standard Runge Kutta fourth order method. See 'trace_particle'
    for an explanation of the arguments. This method uses a fixed step size.
    
    Args:
        mm_per_step: the fixed time step is chosen such that the particle make a step of 'mm_per_step'
            when travelling with its initial speed (Δt = mm_per_step/np.linalg.norm(velocity))
    """
    
    Nmax = 20000
    h = mm_per_step/np.linalg.norm(velocity)
    y = np.array([position[0], position[1], velocity[0], velocity[1]])
    
    N = 1
    positions = np.zeros( (Nmax, 4) )
    positions[0, :] = y
    
    def f(_, y):
        Er, Ez = field(y[0], y[1], *args)
        return np.array([y[2], y[3], EM*Er, EM*Ez])
     
    rmin = -rmax if rmin is None else rmin
    
    # 4th order runge kutta
    while rmin <= y[0] <= rmax and zmin <= y[1] <= zmax:
        k1 = h * f(0.0, y, *args)
        k2 = h * f(0.0, y + 0.5 * k1, *args)
        k3 = h * f(0.0, y + 0.5 * k2, *args)
        k4 = h * f(0.0, y + k3, *args)
        y = y + (k1 + 2*(k2+k3) + k4)/6
        assert N < Nmax
        positions[N, :] = y
        N += 1
     
    return positions[:N]

def _z_to_bounds(z1, z2):
    if z1 < 0 and z2 < 0:
        return (min(z1, z2)-1, 1.0)
    elif z1 > 0 and z2 > 0:
        return (-1.0, max(z1, z2)+1)
    else:
        return (min(z1, z2)-1, max(z1, z2)+1)

class PlaneTracer:
    """A PlaneTracer traces a particle starting from the optical axis to a plane (perpendicular
    to the optical axis) and computes the position and velocity at the intersection point. Useful
    to compute aberration coefficients.
    """
    
    def __init__(self, field, z0, zfinal=None, trace_fun=trace_particle):
        """
        Args:
            field: field function (see solver.py)
            z0: starting z coordinate
            zfinal: coordinate of the target plane with which the intersection will be computed
            trace_fun: tracing method to use (see tracing.trace_particle)
        """
        self.field = field
        self.trace_fun = trace_fun
        self.kwargs = dict()
        self.args = ()
        self.rmax = 100
        self.z0 = z0
        self.zfinal = zfinal if zfinal is not None else z0

    def get_z0(self):
        """Get z0 value"""
        return self.z0
    
    def set_tracer_kwargs(self, **kwargs):
        """Set keyword arguments passed to the tracing method."""
        self.kwargs = kwargs
    
    def set_field_potentials(self, *pots):
        """Set the voltages on the electrodes when the field function is a superposition (see tracing.py).
        The voltages are passed as extra arguments to the field function.
        
        Args:
            *pots: potential to apply on the electrodes
        """
        self.args = pots

    def trace(self, angles, energies, r=None, full=False):
        """Compute a number of intersections with the target plane.

        Args:
            angles: the starting angles with the optical angles
            energies: the starting energies in electronvolts (see 'velocity_vec') 
            r: optionally, do not start at the optical axis but at the given r coordinates

        Returns:
            intersections: intersections as returned by 'plane_intersection'
            mask: when a particle fails to intersect the plane, the corresponding element in mask
                will be False. The corresponding values in intersections will be zero. intersections[mask]
                will give all valid intersections
        """
        zmin, zmax = _z_to_bounds(self.z0, self.zfinal)
        intersections = np.zeros( (angles.size, 4) )
        mask = np.full(angles.size, False)
        assert angles.size == energies.size
        r = np.zeros_like(angles) if r is None else r
        positions = []
         
        for i, (a, e) in enumerate(zip(angles, energies)):
            position = np.array([r[i], self.z0]) 
            velocity = velocity_vec(e, a, direction=self.z0<0)
            p = self.trace_fun(position, velocity, self.field, self.rmax, zmin, zmax, args=self.args, **self.kwargs)

            intersection = plane_intersection(p, self.zfinal)
            positions.append(p)
            
            if intersection is not None:
                intersections[i] = intersection
                mask[i] = True
         
        if not full:
            return intersections, mask
        else:
            return intersections, mask, positions

    def trace_single(self, r, angle, energy):
        """Trace a single particle.

        Args:
            r: starting r coordinate
            angle: starting angle with the optical axis
            energy: initial kinetic energy of the particle in electronvolt
        
        Returns:
            The result of calling 'trace_fun'
        """
        zmin, zmax = _z_to_bounds(self.z0, self.zfinal)
        position = np.array([r, self.z0])
        velocity = velocity_vec(energy, angle, direction=self.z0<0)
         
        return self.trace_fun(position, velocity, self.field, self.rmax, zmin, zmax, args=self.args, **self.kwargs)
    
    def _compute_resulting_angle(self, r, energy):
        zmin, zmax = _z_to_bounds(self.z0, self.zfinal)

        positions = []
        angles = np.zeros_like(r)
         
        for i, r_ in enumerate(r):
            position = np.array([r_, self.z0])
            velocity = velocity_vec(energy, 0.0, direction=self.z0<0)
            p = self.trace_fun(position, velocity, self.field, self.rmax, zmin, zmax, args=self.args, **self.kwargs)
            angles[i] = np.arctan2(p[-1, 0] - p[-2, 0], p[-1, 1] - p[-2, 1])
            positions.append(p)
        
        return angles, positions

    def _benchmark(self, energy=1000, N=100):
        angles = np.full(N, 0.05)
        energies = np.full(N, energy)
         
        # Compile
        _, mask = self.trace(angles, energies)
        start = time.time()
        _, mask = self.trace(angles, energies)
        end = time.time()
        assert np.sum(mask) == N
        print(f'Tracing electron took: {(end-start)/N*1e3:.3f} ms')
        return (end-start)/N


class DoubleTracer:
    
    def __init__(self, field1, field2, z_coords, trace_fun=trace_particle):
        # z coordinate must have four values:
        # - Distance from mirror at previous crossover (crossover 1)
        # - Distance of crossover from first mirror produced by first mirror (crossover 2)
        # - Distance to second mirror at crossover 2
        # - Distance of crossover from second mirror produced by the second mirror
        # Example z_coords = [5, 10, 8, 12]
        # The electron start from a point 5 mm from first mirror.
        # After reflection the beam has a crossover at 10 mm from the first mirror.
        # From this crossover the beam still needs to travel 8 mm to reach the second mirror.
        # The second mirror then projects the beam at 12 mm from the second mirror.
        assert len(z_coords) == 4
        assert all([z > 0 for z in z_coords])
        
        self.z_coords = z_coords
        self.field1, self.field2 = field1, field2
        self.args1, self.args2 = (), ()
        self.rmax = 5
        self.kwargs = dict()
        self.trace_fun = trace_fun

    def get_z0(self):
        return self.z_coords[0]
    
    def set_tracer_kwargs(self, **kwargs):
        self.kwargs = kwargs
    
    def set_field1_potentials(self, *pots):
        self.args1 = pots

    def set_field_potentials(self, *pots):
        self.set_field1_potentials(*pots)
        self.set_field2_potentials(*pots)
    
    def set_field2_potentials(self, *pots):
        self.args2 = pots
     
    def trace(self, angles, energies, r=None):
        assert angles.size == energies.size
        
        intersections = np.zeros( (angles.size, 4) )
        mask = np.full(angles.size, False)
        r = np.zeros_like(angles) if r is None else r
         
        for i, (a, e) in enumerate(zip(angles, energies)):
            position = np.array([r[i], self.z_coords[0]]) 
            velocity = velocity_vec(e, a, direction=self.z_coords[0]<0)
            zmin, zmax = _z_to_bounds(self.z_coords[0], self.z_coords[1])
            p1 = self.trace_fun(position, velocity, self.field1, self.rmax, zmin, zmax, args=self.args1, **self.kwargs)
             
            i1 = plane_intersection(p1, self.z_coords[1])

            if i1 is None:
                continue
            
            assert np.isclose(i1[1], self.z_coords[1])
             
            zmin, zmax = _z_to_bounds(self.z_coords[2], self.z_coords[3])
            position = np.array([i1[0], self.z_coords[2]])
            velocity = np.array([i1[2], -i1[3]]) # Swap around vz value
            p2 = self.trace_fun(position, velocity, self.field2, self.rmax, zmin, zmax, args=self.args2, **self.kwargs)
            
            i2 = plane_intersection(p2, self.z_coords[3])

            if i2 is None:
                continue
             
            assert np.isclose(i2[1], self.z_coords[3])
             
            intersections[i] = i2
            mask[i] = True
             
            '''
            plt.figure()
            plt.axvline(17.65, color='grey', linestyle='dashed',label='Focus plane')
            #plt.axhline(0.075, color='black', label='Electrode boundary')
            plt.plot(p1[:, 1], p1[:, 0], label='First mirror')
            plt.plot(self.z_coords[0] + self.z_coords[1] - p2[:, 1], p2[:, 0], label='Second mirror')
            plt.plot(p2[:, 1], p2[:, 0], label='Second mirror')
            #plt.ylim(-0.15, 0.15)
            plt.xlim(-1, 40)
            plt.xlabel('z (mm)')
            plt.ylabel('r (mm)')
            plt.scatter(intersections[i, 1], intersections[i, 0], color='red')
            plt.scatter(34.645, 0.0, color='grey')
            #plt.axhline(-0.075, color='black')
            plt.legend(loc='upper left')
            plt.savefig(f'images/traces/wrong-trace-vm{self.args1[0]:.3f}-vl{self.args1[1]:.3f}.png')
            plt.show()
            '''
            #print(f'Starting angle: {a:.2e}, first angle: {_angle(*i1[2:]):.2e}, second angle: {_angle(*intersections[i][2:]):.2e}')
        
        return intersections, mask

            

@nb.njit(cache=True, boundscheck=True)
def plane_intersection(positions, z):
    """Calculate the intersection with a plane (perpendicular to the optical axis)
    using a linear interpolation.

    Args:
        positions: trajectory of a particle as returned by 'trace_particle'
        z: z-coordinate of the plane with which the intersection is computed
    
    Returns:
        np.ndarray of shape (4,) containing the r coordinate, z coordinate, velocity
        in r direction, velocity in z direction at the intersection point. Returns None
        if the trajectory does not intersect the plane.
    """
     
    assert len(positions) > 1, "Not enough positions supplied"

    for i in range(len(positions)-1, -1, -1):
        z1 = positions[i-1, 1]
        z2 = positions[i, 1]
        
        if min(z1, z2) <= z <= max(z1, z2):
            ratio = abs(z - z1) / abs(z1 - z2)
            assert 0 <= ratio <= 1
            return positions[i-1] + ratio * (positions[i] - positions[i-1])

    return None

def axis_intersection(positions):
    """Calculate the intersection with the optical axis using a linear interpolation.

    Args:
        positions: trajectory of a particle as returned by 'trace_particle'
     
    Returns:
        np.ndarray of shape (4,) containing the r coordinate, z coordinate, velocity
        in r direction, velocity in z direction at the intersection point. Returns None
        if the trajectory does not intersect the plane.
    """
 
    if positions[-1, 0] <= 0:
        indices = np.where(positions[:, 0] < 0)[0]
    else: 
        indices = np.where(positions[:, 0] > 0)[0]
     
    if not len(indices):
        return None
     
    idx = indices[0]
    ratio = np.abs(positions[idx-1, 0]) / np.abs(positions[idx, 0] - positions[idx-1,0])
    return positions[idx-1, 1] + ratio*(positions[idx, 1] - positions[idx-1, 1])

# Some tests
if __name__ == '__main__':
    p = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0]])
    r = plane_intersection(p, 0.5)[0]
    assert np.isclose(r, 1.0)
    
    p = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0]])
    r = plane_intersection(p, -0.5)[0]
    assert np.isclose(r, 1.0)

    p = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [2.0, -2.0, 0.0, 0.0]])
    r = plane_intersection(p, -1.5)[0]
    assert np.isclose(r, 1.5)
    
    p = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [4.0, 2.0, 0.0, 0.0]])
    r = plane_intersection(p, 1.75)[0]
    assert np.isclose(r, 3.25)








