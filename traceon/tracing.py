from math import sqrt, cos, sin, atan2
import time
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import *

from . import solver as S
from . import backend

EM = -0.1758820022723908 # e/m units ns and mm


def velocity_vec(eV, direction):
    assert eV > 0.0
    
    if eV > 10000:
        print(f'WARNING: velocity vector with large energy ({eV} eV) requested. Note that relativistic tracing is not yet implemented.')
     
    # From electronvolt to mm/ns
    V = 0.5930969604919433*sqrt(eV)
    return V* np.array(direction)/np.linalg.norm(direction)

def velocity_vec_spherical(eV, theta, phi):
    return velocity_vec(eV, [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)])

def velocity_vec_xz_plane(eV, angle, downward=True, three_dimensional=False):
    
    sign = -1 if downward else 1
    direction = [sin(angle), sign*cos(angle)] if not three_dimensional else [sin(angle), 0.0, sign*cos(angle)]
    return velocity_vec(eV, direction)
    
     

def _angle(vr, vz):
    return np.sign(vr) * np.arctan(np.abs(vr/vz))

STEP_MAX = 0.085
STEP_MIN = STEP_MAX/1e10

def trace_particle(position, velocity, field, bounds, rmin=None, args=(), atol=1e-10):
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
    times, positions = _trace_particle(position, velocity, field, bounds, args=args, atol=atol)
    
    if len(times) == 1:
        return times[0], positions[0]
    else:
        return np.concatenate(times, axis=0), np.concatenate(positions, axis=0)

def _z_to_bounds(z1, z2):
    if z1 < 0 and z2 < 0:
        return (min(z1, z2)-1, 1.0)
    elif z1 > 0 and z2 > 0:
        return (-1.0, max(z1, z2)+1)
    else:
        return (min(z1, z2)-1, max(z1, z2)+1)

class Interpolation(Enum):
    NONE = 0,
    AXIAL_DERIVS = 1,

class Tracer:

    def __init__(self, field, bounds, interpolate=Interpolation.NONE, atol=1e-10):
         
        self.geometry = field.geometry
        assert self.geometry.bounds is None or len(bounds) == len(self.geometry.bounds)
        self.field = field
        assert isinstance(field, S.Field)
        
        symmetry = self.geometry.symmetry
        assert (symmetry == '3d' and len(bounds) == 3) or len(bounds) == 2
        self.bounds = bounds
         
        self.interpolate = interpolate
        self.atol = atol
        
        if self.interpolate == Interpolation.AXIAL_DERIVS:
            if symmetry == '3d':
                self.z_axial, self.z_coeff_interpolation = self.field.get_radial_series_coeffs_3d()
            elif symmetry == 'radial':
                self.z_axial, self.z_coeff_interpolation = self.field.get_derivative_interpolation_coeffs()
     
    def __call__(self, position, velocity):
        if self.interpolate == Interpolation.NONE:
            return self._trace_naive(position, velocity)
        elif self.interpolate == Interpolation.AXIAL_DERIVS:
            return self._trace_axial_derivs(position, velocity)

    def _trace_naive(self, position, velocity):
        if self.field.geometry.symmetry == 'radial':
            return backend.trace_particle_radial(position, velocity, self.bounds, self.atol,
                self.field.vertices, self.field.charges)
        elif self.field.geometry.symmetry == '3d':
            return backend.trace_particle_3d(position, velocity, self.bounds, self.atol,
                self.field.vertices, self.field.charges)
     
    def _trace_axial_derivs(self, position, velocity):

        if self.geometry.symmetry == '3d':
            return backend.trace_particle_3d_derivs(position, velocity, self.bounds, self.atol,
                self.z_axial, self.z_coeff_interpolation)
        elif self.geometry.symmetry == 'radial':
            return backend.trace_particle_radial_derivs(position, velocity, self.bounds, self.atol,
                self.z_axial, self.z_coeff_interpolation)
                

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


def xy_plane_intersection(*args, **kwargs):
    return backend.xy_plane_intersection(*args, **kwargs)

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



