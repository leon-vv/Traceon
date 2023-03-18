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

from . import solver as S
from . import backend

EM = -0.1758820022723908 # e/m units ns and mm


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
    Initial velocity vector with magnitude corresponding to the supplied energy and correct units (mm/ns).
    The shape of the resulting vector is the same as the shape of `direction`.
    """
    assert eV > 0.0
    
    if eV > 10000:
        print(f'WARNING: velocity vector with large energy ({eV} eV) requested. Note that relativistic tracing is not yet implemented.')
     
    # From electronvolt to mm/ns
    V = 0.5930969604919433*sqrt(eV)
    return V* np.array(direction)/np.linalg.norm(direction)

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
    Initial velocity vector of shape (3,) with magnitude corresponding to the supplied energy and correct units (mm/ns).
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
    Initial velocity vector with magnitude corresponding to the supplied energy and correct units (mm/ns).
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
    bounds: (2, 3) or (3, 3) np.ndarray of float64
        Once the electron reaches one of the boundaries the tracing stops. The bounds are either of the form
        ( (rmin, rmax), (zmin, zmax) ) for radial symmetric geometries or 
        ( (xmin, xmax), (ymin, ymax), (zmin, zmax) ) for 3D geometries.
    atol: float
        Absolute tolerance determining the accuracy of the trace.
    """
    
    def __init__(self, field, bounds, atol=1e-10):
          
        self.field = field
        assert isinstance(field, S.FieldRadialBEM) or isinstance(field, S.FieldRadialAxial) or \
               isinstance(field, S.Field3D_BEM)    or isinstance(field, S.Field3DAxial)
         
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
            Initial velocity (in units of mm/ns). Use one of the utility functions documented
            above to create the initial velocity vector.
        
        Returns
        -------
        `(times, positions)` which is a tuple of two numpy arrays. `times` is one dimensional and contains the times
        (in ns) at which the positions have been computed. The `positions` array is two dimensional, `positions[i]` correspond
        to time step `times[i]`. One element of the positions array has either shape (4,) in radial symmetry or (6,) in three
        dimensional geometries. The last two or three (depending on symmetry) elements in `positions[i]` contain the corresponding
        velocity vector.
        """

        if isinstance(self.field, S.FieldRadialBEM):
            return backend.trace_particle_radial(position, velocity, self.bounds, self.atol, self.field.vertices, self.field.charges)
        elif isinstance(self.field, S.FieldRadialAxial):
            return backend.trace_particle_radial_derivs(position, velocity, self.bounds, self.atol, self.field.z, self.field.coeffs)
        elif isinstance(self.field, S.Field3D_BEM):
            return backend.trace_particle_3d(position, velocity, self.bounds, self.atol, self.field.vertices, self.field.charges)
        elif isinstance(self.field, S.Field3DAxial):
            return backend.trace_particle_3d_derivs(position, velocity, self.bounds, self.atol, self.field.z, self.field.coeffs)
 

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
    return backend.xy_plane_intersection(positions, z)

def axis_intersection(positions):
    """Calculate the intersection with the optical axis using a linear interpolation. Currently only makes
    sense in radial symmetry, since in a 3D geometry the electron will never pass exactly through the optical axis.

    Parameters
    ----------
    positions: (N, 4) np.ndarray of float64
        positions of an electron as returned by `Tracer`.

    Returns
    ----------
    np.ndarray of shape (4,) containing the r coordinate, z coordinate, velocity in r direction,
    velocity in z direction at the intersection point. Returns None if the trajectory does not intersect the plane.
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



