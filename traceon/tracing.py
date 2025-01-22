"""The tracing module allows to trace charged particles within any field type returned by the `traceon.solver` module. The tracing algorithm
used is RK45 with adaptive step size control [1]. The tracing code is implemented in C and has therefore
excellent performance. The module also provides various helper functions to define appropriate initial velocity vectors and to
compute intersections of the computed traces with various planes.

### References
[1] Erwin Fehlberg. Low-Order Classical Runge-Kutta Formulas With Stepsize Control and their Application to Some Heat
Transfer Problems. 1969. National Aeronautics and Space Administration."""

from __future__ import annotations
import ctypes
from math import sqrt, cos, sin, atan2
import time
from enum import Enum

import numpy as np
import scipy
from scipy.constants import m_e, e, mu_0
    

from . import backend
from . import logging

from ._typing import * 

def velocity_vec(eV: float, direction_: VectorLike3D) -> Vector3D:
    """Compute an initial velocity vector in the correct units and direction.
    
    Parameters
    ----------
    eV: float
        initial energy in units of eV
    direction: (3,) numpy array
        vector giving the correct direction of the initial velocity vector. Does not
        have to be a unit vector as it is always normalized.

    Returns
    -------
    Initial velocity vector with magnitude corresponding to the supplied energy (in eV).
    The shape of the resulting vector is the same as the shape of `direction`.
    """
    assert eV > 0.0, "Please provide a positive energy in eV"

    direction = np.array(direction_)
    assert direction.shape == (3,), "Please provide a three dimensional direction vector"
    
    if eV > 40000:
        logging.log_warning(f'Velocity vector with large energy ({eV} eV) requested. Note that relativistic tracing is not yet implemented.')
    
    return eV * np.array(direction)/np.linalg.norm(direction)

def velocity_vec_spherical(eV: float, theta: float, phi: float) -> Vector3D:
    """Compute initial velocity vector given energy and direction computed from spherical coordinates.
    
    Parameters
    ----------
    eV: float
        initial energy in units of eV
    theta: float
        angle with z-axis (same definition as theta in a spherical coordinate system)
    phi: float
        angle with the x-axis (same definition as phi in a spherical coordinate system)

    Returns
    ------
    Initial velocity vector of shape (3,) with magnitude corresponding to the supplied energy (in eV).
    """
    return velocity_vec(eV, [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)])

def velocity_vec_xz_plane(eV: float, angle: float, downward: bool = True) -> Vector3D:
    """Compute initial velocity vector in the xz plane with the given energy and angle with z-axis.
    
    Parameters
    ----------
    eV: float
        initial energy in units of eV
    angle: float
        angle with z-axis
    downward: bool
        whether the velocity vector should point upward or downwards
     
    Returns
    ------
    Initial velocity vector of shape (3,) with magnitude corresponding to the supplied energy (in eV).
    """
    sign = -1 if downward else 1
    direction = [sin(angle), 0.0, sign*cos(angle)]
    return velocity_vec(eV, direction)
    
def _z_to_bounds(z1: float, z2: float) -> tuple[float, float]:
    if z1 < 0 and z2 < 0:
        return (min(z1, z2)-1, 1.0)
    elif z1 > 0 and z2 > 0:
        return (-1.0, max(z1, z2)+1)
    else:
        return (min(z1, z2)-1, max(z1, z2)+1)

def _convert_velocity_to_SI(velocity: VectorLike3D, mass: float) -> Vector3D:
    # Convert a velocity vector expressed in eV (see functions above)
    # to one expressed in m/s.
    velocity = np.array(velocity)
    speed_eV = np.linalg.norm(velocity)
    speed = sqrt(2*speed_eV*e/mass)
    direction = velocity / speed_eV
    return speed * direction

class Tracer:
    """General tracer class for charged particles. Can trace charged particles given any field class from `traceon.solver`.

    Parameters
    ----------
    field: `traceon.field.Field`
        The field used to compute the force felt by the charged particle. Note that any child class of `traceon.field.Field` can be used.
    bounds: (3, 2) np.ndarray of float64
        Once the particle reaches one of the boundaries the tracing stops. The bounds are of the form ( (xmin, xmax), (ymin, ymax), (zmin, zmax) ).
    """
    
    def __init__(self, field: Field, bounds: Bounds3D) -> None:
        self.field = field
        bounds = np.array(bounds).astype(np.float64)
        assert bounds.shape == (3,2)
        self.bounds = bounds

        self.trace_fun, self.low_level_args, *keep_alive = field.get_low_level_trace_function()
        
        # Allow functions to optionally return references to objects that need
        # to be kept alive as long as the tracer is kept alive. This prevents
        # memory from being reclaimed while the C backend is still working with it.
        self.keep_alive = keep_alive

        if self.low_level_args is None:
            self.trace_args = None
        elif isinstance(self.low_level_args, int): # Interpret as literal memory address
            self.trace_args = ctypes.c_void_p(self.low_level_args)
        else: # Interpret as anything ctypes can make sense of
            self.trace_args = ctypes.cast(ctypes.pointer(self.low_level_args), ctypes.c_void_p)
     
    def __str__(self) -> str:
        field_name = self.field.__class__.__name__
        bounds_str = ' '.join([f'({bmin:.2f}, {bmax:.2f})' for bmin, bmax in self.bounds])
        return f'<Traceon Tracer of {field_name},\n\t' \
            + 'Bounds: ' + bounds_str + ' mm >'
    
    def __call__(self, 
                 position: PointLike3D, 
                 velocity: VectorLike3D, 
                 mass: float = m_e, 
                 charge: float = -e, 
                 atol: float = 1e-8) -> tuple[ArrayFloat1D, ArrayFloat2D]:
        """Trace a charged particle.

        Parameters
        ----------
        position: (3,) np.ndarray of float64
            Initial position of the particle.
        velocity: (3,) np.ndarray of float64
            Initial velocity (expressed in a vector whose magnitude has units of eV). Use one of the utility functions documented
            above to create the initial velocity vector.
        mass: float
            Particle mass in kilogram (kg). The default value is the electron mass: m_e = 9.1093837015e-31 kg.
        charge: float
            Particle charge in Coulomb (C). The default value is the electron charge: -1 * e = -1.602176634e-19 C.
        atol: float
            Absolute tolerance determining the accuracy of the trace.
        
        Returns
        -------
        `(times, positions)` which is a tuple of two numpy arrays. `times` is one dimensional and contains the times
        at which the positions have been computed. The `positions` array is two dimensional, `positions[i]` correspond
        to time step `times[i]`. One element of the positions array has shape (6,).
        The first three elements in the `positions[i]` array contain the x,y,z positions.
        The last three elements in `positions[i]` contain the vx,vy,vz velocities.
        """
        position = np.array(position)
        charge_over_mass = charge / mass
        velocity = _convert_velocity_to_SI(velocity, mass)

        return backend.trace_particle(
                position,
                velocity,
                charge_over_mass, 
                self.trace_fun, #type:ignore
                self.bounds,
                atol,
                self.trace_args)

def plane_intersection(positions: ArrayLikeFloat2D, p0: PointLike3D, normal: VectorLike3D) -> ArrayFloat1D:
    """Compute the intersection of a trajectory with a general plane in 3D. The plane is specified
    by a point (p0) in the plane and a normal vector (normal) to the plane. The intersection
    point is calculated using a linear interpolation.
    
    Parameters
    ----------
    positions: (N, 6) np.ndarray of float64
        Positions of a charged particle as returned by `Tracer`.
    
    p0: (3,) np.ndarray of float64
        A point that lies in the plane.

    normal: (3,) np.ndarray of float64
        A vector that is normal to the plane. A point p lies in the plane iff `dot(normal, p - p0) = 0` where
        dot is the dot product.
    
    Returns
    --------
    np.ndarray of shape (6,) containing the position and velocity of the particle at the intersection point.
    """
    positions, p0, normal = np.array(positions, dtype=np.float64), np.array(p0, dtype=np.float64), np.array(normal, dtype=np.float64)

    assert positions.shape == (len(positions), 6), "The positions array should have shape (N, 6)"
    return backend.plane_intersection(positions, p0, normal)

def xy_plane_intersection(positions: ArrayLikeFloat2D, z: float) -> ArrayFloat1D:
    """Compute the intersection of a trajectory with an xy-plane.

    Parameters
    ----------
    positions: (N, 6) np.ndarray of float64
        Positions (and velocities) of a charged particle as returned by `Tracer`.
    z: float
        z-coordinate of the plane with which to compute the intersection
    
    Returns
    --------
    (6,) array of float64, containing the position and velocity of the particle at the intersection point.
    """
    return plane_intersection(positions, [0, 0, z], [0, 0, 1])

def xz_plane_intersection(positions: ArrayLikeFloat2D, y: float) -> ArrayLikeFloat1D:
    """Compute the intersection of a trajectory with an xz-plane.

    Parameters
    ----------
    positions: (N, 6) np.ndarray of float64
        Positions (and velocities) of a charged particle as returned by `Tracer`.
    z: float
        z-coordinate of the plane with which to compute the intersection
    
    Returns
    --------
    (6,) array of float64, containing the position and velocity of the particle at the intersection point.
    """
    return plane_intersection(positions, [0, y, 0], [0, 1, 0])

def yz_plane_intersection(positions: ArrayLikeFloat2D, x: float) -> ArrayLikeFloat1D:
    """Compute the intersection of a trajectory with an yz-plane.

    Parameters
    ----------
    positions: (N, 6) np.ndarray of float64
        Positions (and velocities) of a charged particle as returned by `Tracer`.
    z: float
        z-coordinate of the plane with which to compute the intersection
    
    Returns
    --------
    (6,) array of float64, containing the position and velocity of the particle at the intersection point.
    """
    return plane_intersection(positions, [x, 0, 0], [1, 0, 0])

def axis_intersection(positions: ArrayLikeFloat2D) -> float:
    """Compute the z-value of the intersection of the trajectory with the x=0 plane.
    Note that this function will not work properly if the trajectory crosses the x=0 plane zero or multiple times.
    
    Parameters
    ----------
    positions: (N, 6) np.ndarray of float64
        Positions (and velocities) of a charged particle as returned by `Tracer`.
    
    Returns
    --------
    float, z-value of the intersection with the x=0 plane
    """
    return yz_plane_intersection(positions, 0)[2]

