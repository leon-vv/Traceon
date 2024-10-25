"""The tracing module allows to trace electrons within any field type returned by the `traceon.solver` module. The tracing algorithm
used is RK45 with adaptive step size control [1]. The tracing code is implemented in C (see `traceon.backend`) and has therefore
excellent performance. The module also provides various helper functions to define appropriate initial velocity vectors and to
compute intersections of the computed traces with various planes.

### References
[1] Erwin Fehlberg. Low-Order Classical Runge-Kutta Formulas With Stepsize Control and their Application to Some Heat
Transfer Problems. 1969. National Aeronautics and Space Administration."""


from math import sqrt, cos, sin, atan2
import time
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import *
from scipy.constants import m_e, e

from . import solver as S
from . import backend
from . import logging

def velocity_vec(eV, direction):
    """Compute an initial velocity vector in the correct units and direction.
    
    Parameters
    ----------
    eV: float
        initial energy in units of eV
    direction: (2,) or (3,) numpy array
        vector giving the correct direction of the initial velocity vector. Does not
        have to be a unit vector as it is always normalized.

    Returns
    -------
    Initial velocity vector with magnitude corresponding to the supplied energy (in eV).
    The shape of the resulting vector is the same as the shape of `direction`.
    """
    assert eV > 0.0
    
    if eV > 40000:
        logging.log_warning(f'Velocity vector with large energy ({eV} eV) requested. Note that relativistic tracing is not yet implemented.')
    
    return eV * np.array(direction)/np.linalg.norm(direction)

def velocity_vec_spherical(eV, theta, phi):
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

def velocity_vec_xz_plane(eV, angle, downward=True, three_dimensional=False):
    """Compute initial velocity vector in the xz plane with the given energy and angle with z-axis.
    
    Parameters
    ----------
    eV: float
        initial energy in units of eV
    angle: float
        angle with z-axis
    downward: bool
        whether the velocity vector should point upward or downwards
    three_dimensional: bool
        whether the resulting velocity vector has shape (2,) or shape (3,)
     
    Returns
    ------
    Initial velocity vector with magnitude corresponding to the supplied energy (in eV).
    """
    sign = -1 if downward else 1
    direction = [sin(angle), sign*cos(angle)] if not three_dimensional else [sin(angle), 0.0, sign*cos(angle)]
    return velocity_vec(eV, direction)
    
def _z_to_bounds(z1, z2):
    if z1 < 0 and z2 < 0:
        return (min(z1, z2)-1, 1.0)
    elif z1 > 0 and z2 > 0:
        return (-1.0, max(z1, z2)+1)
    else:
        return (min(z1, z2)-1, max(z1, z2)+1)

class Tracer:
    """General electron tracer class. Can trace electrons given any field class from `traceon.solver`.

    Parameters
    ----------
    field: traceon.solver.Field (or any class inheriting Field)
        The field used to compute the force felt by the electron.
    bounds: (3, 2) np.ndarray of float64
        Once the electron reaches one of the boundaries the tracing stops. The bounds are of the form ( (xmin, xmax), (ymin, ymax), (zmin, zmax) ).
    atol: float
        Absolute tolerance determining the accuracy of the trace.
    """
    
    def __init__(self, field, bounds, atol=1e-10):
          
        self.field = field
        assert isinstance(field, S.FieldRadialBEM) or isinstance(field, S.FieldRadialAxial) or \
               isinstance(field, S.Field3D_BEM)    or isinstance(field, S.Field3DAxial)
         
        bounds = np.array(bounds).astype(np.float64)
        assert bounds.shape == (3,2)
        self.bounds = bounds
        self.atol = atol
    
    def __str__(self):
        field_name = self.field.__class__.__name__
        bounds_str = ' '.join([f'({bmin:.2f}, {bmax:.2f})' for bmin, bmax in self.bounds])
        return f'<Traceon Tracer of {field_name},\n\t' \
            + 'Bounds: ' + bounds_str + ' mm >'
        
    def __call__(self, position, velocity):
        """Trace an electron.

        Parameters
        ----------
        position: (2,) or (3,) np.ndarray of float64
            Initial position of electron.
        velocity: (2,) or (3,) np.ndarray of float64
            Initial velocity (expressed in a vector whose magnitude has units of eV). Use one of the utility functions documented
            above to create the initial velocity vector.
        
        Returns
        -------
        `(times, positions)` which is a tuple of two numpy arrays. `times` is one dimensional and contains the times
        (in ns) at which the positions have been computed. The `positions` array is two dimensional, `positions[i]` correspond
        to time step `times[i]`. One element of the positions array has shape (6,).
        The first three elements in the `positions[i]` array contain the x,y,z positions.
        The last three elements in `positions[i]` contain the vx,vy,vz velocities.
        """
         
        f = self.field
         
        # Convert the velocity in eV to m/s
        speed_eV = np.linalg.norm(velocity)
        speed = sqrt(2*speed_eV*e/m_e)
        direction = velocity / speed_eV
        velocity = speed * direction
        
        if isinstance(self.field, S.FieldRadialBEM):
            return backend.trace_particle_radial(position, velocity, self.bounds, self.atol, 
                f.electrostatic_point_charges, f.magnetostatic_point_charges, f.current_point_charges, field_bounds=f.field_bounds)
        elif isinstance(self.field, S.FieldRadialAxial):
            elec, mag = self.field.electrostatic_coeffs, self.field.magnetostatic_coeffs
            return backend.trace_particle_radial_derivs(position, velocity, self.bounds, self.atol, self.field.z, elec, mag)
        elif isinstance(self.field, S.Field3D_BEM):
            bounds = self.field.field_bounds
            elec, mag = self.field.electrostatic_point_charges, self.field.magnetostatic_point_charges
            return backend.trace_particle_3d(position, velocity, self.bounds, self.atol, elec, mag)
        elif isinstance(self.field, S.Field3DAxial):
            return backend.trace_particle_3d_derivs(position, velocity, self.bounds, self.atol,
                    self.field.z, self.field.electrostatic_coeffs, self.field.magnetostatic_coeffs)
 

def plane_intersection(positions, p0, normal):
    """Compute the intersection of a trajectory with a general plane in 3D. The plane is specified
    by a point (p0) in the plane and a normal vector (normal) to the plane. The intersection
    point is calculated using a linear interpolation.
    
    Parameters
    ----------
    positions: (N, 6) np.ndarray of float64
        Positions of an electron as returned by `Tracer`.
    
    p0: (3,) np.ndarray of float64
        A point that lies in the plane.

    normal: (3,) np.ndarray of float64
        A vector that is normal to the plane. A point p lies in the plane iff `dot(normal, p - p0) = 0` where
        dot is the dot product.
    
    Returns
    --------
    np.ndarray of shape (6,) containing the position and velocity of the electron at the intersection point.
    """

    return backend.plane_intersection(positions, p0, normal)

def line_intersection(positions, p0, tangent):
    """Compute the intersection of a trajectory with a line in 2D. The line is specified
    by a point (p0) on the line and a vector tangential (tangent) to the line. The intersection
    point is calculated using a linear interpolation.
    
    Parameters
    ----------
    positions: (N, 4) np.ndarray of float64
        Positions of an electron as returned by `Tracer`.
    
    p0: (2,) np.ndarray of float64
        A point that lies on the line.
    
    tangent: (2,) np.ndarray of float64
        A vector that is tangential to the line. A point p lies on the line if there exists a number k
        such that `p0 + k*tangent = p`.
    
    Returns
    --------
    np.ndarray of shape (4,) containing the position and velocity of the electron at the intersection point.
    """

    return backend.line_intersection(positions, p0, tangent)

def xy_plane_intersection(positions, z):
    """Compute the intersection of a trajectory with an xy-plane.

    Parameters
    ----------
    positions: (N, 4) or (N, 6) np.ndarray of float64
        Positions of an electron as returned by `Tracer`.
    z: float
        z-coordinate of the plane with which to compute the intersection
    
    Returns
    --------
    np.ndarray of shape (4,) or (6,) containing the position and velocity of the electron at the intersection point.
    """
    assert positions.shape == (len(positions), 4) or positions.shape == (len(positions), 6)
    
    if positions.shape[1] == 4:
        return line_intersection(positions, np.array([0., z]), np.array([1.0, 0.0]))
    else:
        return plane_intersection(positions, np.array([0.,0.,z]), np.array([0., 0., 1.0]))

def xz_plane_intersection(positions, y):
    """Compute the intersection of a trajectory with an xz-plane. Note that this function
    does not make sense in 2D (where we consider (r,z) as (x,z) and therefore the y-axis is missing).

    Parameters
    ----------
    positions: (N, 6) np.ndarray of float64
        Positions of an electron as returned by `Tracer`.
    y: float
        y-coordinate of the plane with which to compute the intersection
    
    Returns
    --------
    np.ndarray of shape (6,) containing the position and velocity of the electron at the intersection point.
    """
    return plane_intersection(positions, np.array([0.,y,0.]), np.array([0., 1.0, 0.]))

def yz_plane_intersection(positions, x):
    """Compute the intersection of a trajectory with an yz-plane.

    Parameters
    ----------
    positions: (N, 4) or (N, 6) np.ndarray of float64
        Positions of an electron as returned by `Tracer`.
    x: float
        x-coordinate of the plane with which to compute the intersection
    
    Returns
    --------
    np.ndarray of shape (4,) or (6,) containing the position and velocity of the electron at the intersection point.
    """
    assert positions.shape == (len(positions), 4) or positions.shape == (len(positions), 6)
     
    if positions.shape[1] == 4:
        return line_intersection(positions, np.array([x, 0.]), np.array([0.0, 1.0]))
    else:
        return plane_intersection(positions, np.array([x,0.,0.]), np.array([1.0, 0., 0.]))


def axis_intersection(positions):
    """Calculate the intersection with the optical axis using a linear interpolation. Notice that
    this only makes sense in 2D as in 3D the particle will never pass exactly through the optical axis.
    However, this function is implemented as `yz_plane_intersection(positions, 0.0)` and will therefore
    give meaningful results in 3D if you expect the particle trajectory to be in the xz plane. This function
    only returns the z-coordinate. Use `yz_plane_intersection` directly if you want to retrieve the velocity 
    components.
    
    Parameters
    ----------
    positions: (N, 4) or (N, 6) np.ndarray of float64
        positions of an electron as returned by `Tracer`.
    
    Returns
    ----------
    float z-coordinate of intersection point
    """
    assert positions.shape == (len(positions), 4) or positions.shape == (len(positions), 6)
    
    z_index = 1 if positions.shape[1] == 4 else 2
    return yz_plane_intersection(positions, 0.)[z_index]



