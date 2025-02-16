"""
Module containing a single function to find the focus of a beam of electron trajecories.
"""
from __future__ import annotations
import numpy as np

from .typing import * 

def focus_position(trajectories: list[Path]) -> Point3D:
    """
    Find the focus of the given trajectories (which are returned from `voltrace.tracing.Tracer.__call__`).
    The focus is found using a least square method by considering the final positions and velocities of
    the given trajectories and linearly extending the trajectories backwards.
     
    
    Parameters
    ------------
    trajectories: list[Path]
        Trajectories of particles, as returned by `voltrace.tracing.Tracer.__call__`
     
    Returns
    --------------
    (3,) np.ndarray of float64, representing the position of the focus
    """
    final_positions = np.array([t.endpoint() for t in trajectories])
    final_velocities = np.array([t.velocity_vector(t.parameter_range) for t in trajectories])

    angles_x = np.array([v[0]/v[2] for p, v in zip(final_positions, final_velocities)])
    angles_y = np.array([v[1]/v[2] for p, v in zip(final_positions, final_velocities)])
    x, y, z = final_positions.T
    
    N = len(trajectories)
    first_column = np.concatenate( (-angles_x, -angles_y) )
    second_column = np.concatenate( (np.ones(N), np.zeros(N)) )
    third_column = np.concatenate( (np.zeros(N), np.ones(N)) )
    right_hand_side = np.concatenate( (x - z*angles_x, y - z*angles_y) )
     
    (z, x, y) = np.linalg.lstsq(
        np.array([first_column, second_column, third_column]).T,
        right_hand_side, rcond=None)[0]
     
    return np.array([x, y, z])
    
