from math import sqrt, cos, sin, atan2
import time

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import scipy
from scipy.integrate import *

from . import solver as S
from .util import traceon_jit

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
    times, positions = _trace_particle(position, velocity, field, rmax, zmin, zmax, rmin=rmin, args=args, atol=atol)
    
    if len(times) == 1:
        return times[0], positions[0]
    else:
        return np.concatenate(times, axis=0), np.concatenate(positions, axis=0)

@traceon_jit(inline='always')
def _field_to_acceleration(field, args, y):
    if y.shape == (4,):
        Er, Ez = field(y, *args)
        return np.array([y[2], y[3], EM*Er, EM*Ez])
    elif y.shape == (6,):
        Ex, Ey, Ez = field(y, *args)
        return np.array([y[3], y[4], y[5], EM*Ex, EM*Ey, EM*Ez])

@nb.njit(fastmath=True)
def _trace_particle(position, velocity, field, rmax, zmin, zmax, rmin=None, args=(), atol=1e-10):
    Nblock = int(1e5)
    V = np.linalg.norm(velocity)
    h = STEP_MAX/V
    hmax = STEP_MAX/V
    hmin = STEP_MIN/V
     
    y = np.array([*position, *velocity])
     
    position_block = np.zeros( (Nblock, y.size) ) # Could be np.empty?
    positions = [position_block]
    position_block[0, :] = y
     
    time_block = np.zeros(Nblock)
    times = [time_block]
    time_block[0] = 0.0
    last_time = 0.0
    
    N = 1
    
    A = (0.0, 0.0, 2/9, 1/3, 3/4, 1, 5/6) # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    B6 = (0.0, 65/432, -5/16, 13/16, 4/27, 5/144) # Left pad with 0.0 to keep indices the same
    B5 = (0.0, -17/12, 27/4, -27/5, 16/15)
    B4 = (0.0, 69/128, -243/128, 135/64)
    B3 = (0.0, 1/12, 1/4)
    B2 = (0.0, 2/9)
    CH = (0.0, 47/450, 0, 12/25, 32/225, 1/30, 6/25)
    CT = (0.0, -1/150, 0, 3/100, -16/75, -1/20, 6/25)

    rmin = -rmax if rmin is None else rmin
     
    while rmin <= y[0] <= rmax and (position.size==3 or zmin <= y[1] <= zmax) and (position.size==2 or zmin <= y[2] <= zmax):
        k1 = h * _field_to_acceleration(field, args,y)
        k2 = h * _field_to_acceleration(field, args,y + B2[1]*k1)
        k3 = h * _field_to_acceleration(field, args,y + B3[1]*k1 + B3[2]*k2)
        k4 = h * _field_to_acceleration(field, args,y + B4[1]*k1 + B4[2]*k2 + B4[3]*k3)
        k5 = h * _field_to_acceleration(field, args,y + B5[1]*k1 + B5[2]*k2 + B5[3]*k3 + B5[4]*k4)
        k6 = h * _field_to_acceleration(field, args,y + B6[1]*k1 + B6[2]*k2 + B6[3]*k3 + B6[4]*k4 + B6[5]*k5)
        
        TE = np.max(np.abs(CT[1]*k1 + CT[2]*k2 + CT[3]*k3 + CT[4]*k4 + CT[5]*k5 + CT[6]*k6))
         
        if TE <= atol or h == hmin:
            y = y + CH[1]*k1 + CH[2]*k2 + CH[3]*k3 + CH[4]*k4 + CH[5]*k5 + CH[6]*k6

            if N == Nblock:
                position_block = np.zeros( (Nblock, y.size) )
                positions.append(position_block)
                times_block = np.zeros(Nblock)
                times.append(times_block)
                N = 0
             
            position_block[N, :] = y
            last_time += h
            time_block[N] = last_time
            N += 1
         
        if TE > atol/10:
            h = max(min(0.9*h*(atol/TE)**(1/5), hmax), hmin)
        elif TE < atol/100:
            h = hmax
     
    positions[-1] = positions[-1][:N]
    times[-1] = times[-1][:N]
    return times, positions

def _z_to_bounds(z1, z2):
    if z1 < 0 and z2 < 0:
        return (min(z1, z2)-1, 1.0)
    elif z1 > 0 and z2 > 0:
        return (-1.0, max(z1, z2)+1)
    else:
        return (min(z1, z2)-1, max(z1, z2)+1)


class Tracer:

    def __init__(self, field, rmax, zmin, zmax, rmin=None, interpolate=True, atol=1e-10):
         
        self.field = field
        assert isinstance(field, S.Field) or isinstance(field, S.FieldSuperposition)
        self.rmin = rmin
        self.rmax = rmax
        self.zmin = zmin
        self.zmax = zmax
        self.interpolate = interpolate
        self.atol = atol
    
    def __call__(self, position, velocity):
        if self.interpolate:
            return self._trace_interpolated(position, velocity)
        else:
            return self._trace_naive(position, velocity)
        

    def _trace_naive(self, position, velocity):
     
        if isinstance(self.field, S.Field):

            args = (self.field.geometry.symmetry, self.field.vertices, self.field.charges)
            
            return trace_particle(position, velocity,
                S._field_at_point,
                self.rmax, self.zmin, self.zmax, rmin=self.rmin, args=args, atol=self.atol)

        elif isinstance(self.field, S.FieldSuperposition):

            symmetries = [f.geometry.symmetry for f in self.field.fields]
            lines = [f.vertices for f in self.field.fields]
            charges = [f.charges for f in self.field.fields]

            return trace_particle(position, velocity,
                S._field_at_point_superposition,
                self.rmax, self.zmin, self.zmax, args=(self.field.scales, symmetries, lines, charges),
                atol=self.atol, rmin=self.rmin)
      
    
    def _trace_interpolated(self, position, velocity):
        assert self.field.geometry.symmetry == 'radial'
        
        z, coeffs = self.field.get_derivative_interpolation_coeffs()
         
        return trace_particle(position, velocity,
            S._field_from_interpolated_derivatives, 
            self.rmax, self.zmin, self.zmax, args=(z, coeffs),
            atol=self.atol, rmin=self.rmin)
        

class PlaneTracer:
    """A PlaneTracer traces a particle starting from the optical axis to a plane (perpendicular
    to the optical axis) and computes the position and velocity at the intersection point. Useful
    to compute aberration coefficients.
    """
    
    def __init__(self, field, z0, interpolate=True, rmax=100, zfinal=None):
        """
        Args:
            field: field function (see solver.py)
            z0: starting z coordinate
            zfinal: coordinate of the target plane with which the intersection will be computed
            trace_fun: tracing method to use (see tracing.trace_particle)
        """
        self.field = field
        self.kwargs = dict()
        self.rmax = rmax
        self.z0 = z0
        self.zfinal = zfinal if zfinal is not None else z0
        self.interpolate=interpolate
    
    def set_tracer_kwargs(self, **kwargs):
        """Set keyword arguments passed to the tracing method."""
        self.kwargs = kwargs

    def get_z0(self):
        return self.z0
     
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
        tracer = Tracer(self.field, self.rmax, zmin, zmax, interpolate=self.interpolate)
        
        intersections = np.zeros( (angles.size, 4) )
        mask = np.full(angles.size, False)
        assert angles.size == energies.size
        r = np.zeros_like(angles) if r is None else r
        positions = []
         
        for i, (a, e) in enumerate(zip(angles, energies)):
            position = np.array([r[i], self.z0]) 
            velocity = velocity_vec(e, a, direction=self.z0<0)
            _, p = tracer(position, velocity)
            
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
        z1 = positions[i-1, 1 if positions.shape[1] == 4 else 2]
        z2 = positions[i, 1 if positions.shape[1] == 4 else 2]
        
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



